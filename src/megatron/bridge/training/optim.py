# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Optional, Union

import torch

from megatron.core.optimizer import (
    MegatronOptimizer,
    OptimizerConfig,
    get_megatron_optimizer,
)
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.training.config import (
    OptimizerConfigOverrideProvider,
    OptimizerConfigOverrideProviderContext,
    SchedulerConfig,
)


_LOG = logging.getLogger(__name__)


def setup_optimizer(
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    model: Union[MegatronModule, list[MegatronModule]],
    use_gloo_process_groups: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
    optimizer_config_override_provider: Optional[OptimizerConfigOverrideProvider] = None,
) -> tuple[MegatronOptimizer, OptimizerParamScheduler]:
    """Set up the optimizer and scheduler.

    Args:
        optimizer_config: Configuration for the optimizer
        scheduler_config: Configuration for the scheduler
        model: The model to optimize
        use_gloo_process_groups: Whether to use Gloo process groups
        pg_collection: Optional process group collection for distributed training

    Returns:
        tuple containing the optimizer and scheduler
    """
    if optimizer_config_override_provider is None:
        optimizer_config_override_provider = OptimizerConfigOverrideProvider()

    # Build config overrides for weight decay based on scheduler config and model params
    config_overrides = optimizer_config_override_provider.build_config_overrides(
        OptimizerConfigOverrideProviderContext(scheduler_config, optimizer_config, model)
    )

    if "muon" not in optimizer_config.optimizer and "soap" not in optimizer_config.optimizer:
        optimizer = get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=model,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
            pg_collection=pg_collection,
        )
    else:
        optimizer = get_megatron_muon_optimizer(
            config=optimizer_config,
            model_chunks=model,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
            layer_wise_distributed_optimizer="dist" in optimizer_config.optimizer,
            pg_collection=pg_collection,
        )

    scheduler = _get_scheduler(optimizer_config, scheduler_config, optimizer)

    optimizer = _maybe_wrap_with_cautious_wd(optimizer, optimizer_config)

    return optimizer, scheduler


def _maybe_wrap_with_cautious_wd(
    optimizer: MegatronOptimizer, optimizer_config: OptimizerConfig
) -> MegatronOptimizer:
    """Wrap ``optimizer.step()`` with Cautious Weight Decay (modded-nanogpt PR #154).

    Activated by setting the ``BUCKET_A_CAUTIOUS_WD`` environment variable to a
    positive float. The factor is applied as ``p -= mask * lr * factor * p`` after
    the optimizer's underlying step, where ``mask = (|p_new| > |p_old|)`` — i.e.
    weight decay is applied only to parameters whose magnitude grew during the step.

    The caller is expected to set ``optimizer_config.weight_decay = 0.0`` so the
    underlying optimizer does not also apply standard WD; otherwise both compound.

    The wrapped step also propagates the post-CWD master parameters back to the
    bf16/fp16 model copies via ``_copy_main_params_to_model_params`` if available
    on the optimizer (DistributedOptimizer / MixedPrecisionOptimizer).
    """
    factor_str = os.environ.get("BUCKET_A_CAUTIOUS_WD", "")
    if not factor_str:
        return optimizer
    try:
        factor = float(factor_str)
    except ValueError:
        _LOG.warning("BUCKET_A_CAUTIOUS_WD=%r is not a float; ignoring", factor_str)
        return optimizer
    if factor <= 0:
        return optimizer

    if getattr(optimizer_config, "weight_decay", 0.0) > 0:
        _LOG.warning(
            "BUCKET_A_CAUTIOUS_WD=%s with optimizer_config.weight_decay=%s (>0); "
            "standard WD will compound with cautious WD. Set weight_decay=0 to "
            "use cautious WD only.",
            factor,
            optimizer_config.weight_decay,
        )

    _LOG.info("Cautious Weight Decay wrapper enabled with factor=%s", factor)

    original_step = optimizer.step
    propagate_method = None
    for name in ("_copy_main_params_to_model_params", "reload_model_params"):
        if hasattr(optimizer, name):
            propagate_method = name
            break

    def cautious_step(*args, **kwargs):
        snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad and p.numel() > 0:
                    snapshots.append((p, p.detach().abs().clone()))

        result = original_step(*args, **kwargs)

        lr = float(optimizer.param_groups[0].get("lr", 0.0))
        coef = -lr * factor
        if coef != 0.0:
            with torch.no_grad():
                for p, prev_abs in snapshots:
                    mask = (p.detach().abs() > prev_abs).to(p.dtype)
                    p.data.addcmul_(mask, p.data, value=coef)
            if propagate_method is not None:
                getattr(optimizer, propagate_method)()

        return result

    optimizer.step = cautious_step
    return optimizer


def _get_scheduler(
    optimizer_config: OptimizerConfig, scheduler_config: SchedulerConfig, optimizer: MegatronOptimizer
) -> OptimizerParamScheduler:
    """Get the optimizer parameter scheduler.

    Args:
        optimizer_config: Configuration for the optimizer
        scheduler_config: Configuration for the scheduler
        optimizer: The optimizer to schedule

    Returns:
        The optimizer parameter scheduler
    """
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=scheduler_config.lr_warmup_init,
        max_lr=optimizer_config.lr,
        min_lr=optimizer_config.min_lr,
        lr_warmup_steps=scheduler_config.lr_warmup_steps,
        lr_decay_steps=scheduler_config.lr_decay_steps,
        lr_decay_style=scheduler_config.lr_decay_style,
        start_wd=scheduler_config.start_weight_decay,
        end_wd=scheduler_config.end_weight_decay,
        wd_incr_steps=scheduler_config.wd_incr_steps,
        wd_incr_style=scheduler_config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=scheduler_config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=scheduler_config.override_opt_param_scheduler,
        wsd_decay_steps=scheduler_config.wsd_decay_steps,
        lr_wsd_decay_style=scheduler_config.lr_wsd_decay_style,
    )

    return scheduler
