# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""SharedOuterGroupedExpertAdapter with separate gate and up adapters on fc1.

The fc1 adapter must act as two rank-``dim`` adapters, gate and up, each with
its own shared hidden-side factor; its off-diagonal blocks must stay zero; and
``B @ A`` must be exactly the two deltas, so merge and export carry it as is.
"""

import datetime
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.bridge.peft.lora import LoRA, LoRAMerge
from megatron.bridge.peft.utils import SharedOuterGroupedExpertAdapter


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="torch._grouped_mm needs CUDA")

H, INTER, R, E = 32, 16, 8, 3  # hidden, intermediate per half, rank per half, local experts
ALPHA = 16
SPLITS = [5, 0, 7]  # tokens per expert, one expert empty


@pytest.fixture(autouse=True, scope="module")
def model_parallel():
    # tear down only what this module set up: later modules must find the state they left
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29511")
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", world_size=1, rank=0, timeout=datetime.timedelta(minutes=5))
    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    if owns_model_parallel:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)  # expert linears init under the expert-parallel RNG
    yield
    if owns_model_parallel:
        parallel_state.destroy_model_parallel()
    if owns_process_group:
        dist.destroy_process_group()


def _fc1(**overrides) -> SharedOuterGroupedExpertAdapter:
    config = ModelParallelConfig(bf16=True, params_dtype=torch.bfloat16, expert_tensor_parallel_size=1)
    for key, value in overrides.pop("config", {}).items():
        setattr(config, key, value)
    kwargs = dict(
        num_local_experts=E,
        base_linear_name="decoder.layers.0.mlp.experts.linear_fc1",
        activation="identity",
        input_is_parallel=False,
        model_parallel_config=config,
        alpha=ALPHA,
        params_device=torch.device("cuda"),
        params_dtype=torch.bfloat16,
        separate_gate_up=True,
    )
    kwargs.update(overrides)
    return SharedOuterGroupedExpertAdapter(H, 2 * INTER, R, **kwargs).cuda()


def _randomize_diagonal(adapter: SharedOuterGroupedExpertAdapter) -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        adapter.linear_in.weight.copy_(torch.randn(2 * R, H, generator=g, device="cuda"))
        w = adapter.linear_out.weight
        w[:, :INTER, :R].copy_(torch.randn(E, INTER, R, generator=g, device="cuda"))
        w[:, INTER:, R:].copy_(torch.randn(E, INTER, R, generator=g, device="cuda"))


def test_layout_and_scale():
    adapter = _fc1()
    assert adapter.dim == R
    assert tuple(adapter.linear_in.weight.shape) == (2 * R, H)
    assert tuple(adapter.linear_out.weight.shape) == (E, 2 * INTER, 2 * R)
    w = adapter.linear_out.weight
    assert torch.count_nonzero(w[:, :INTER, R:]) == 0 and torch.count_nonzero(w[:, INTER:, :R]) == 0


def test_forward_is_two_independent_adapters_per_expert():
    adapter = _fc1()
    _randomize_diagonal(adapter)
    x = torch.randn(sum(SPLITS), H, device="cuda", dtype=torch.bfloat16)
    out = adapter(x, m_splits=torch.tensor(SPLITS))

    a = adapter.linear_in.weight.float()
    a_gate, a_up = a[:R], a[R:]
    w = adapter.linear_out.weight.float()
    rows, start = [], 0
    for e, n in enumerate(SPLITS):
        xe = x[start : start + n].float()
        # the adapter holds the rank-space activation in bf16 between its two GEMMs
        gate = (xe @ a_gate.T).bfloat16().float() @ w[e, :INTER, :R].T
        up = (xe @ a_up.T).bfloat16().float() @ w[e, INTER:, R:].T
        rows.append(torch.cat([gate, up], dim=-1))
        start += n
    expected = torch.cat(rows) * (ALPHA / R)
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


def test_off_diagonal_blocks_receive_no_gradient():
    adapter = _fc1()
    _randomize_diagonal(adapter)
    x = torch.randn(sum(SPLITS), H, device="cuda", dtype=torch.bfloat16)
    adapter(x, m_splits=torch.tensor(SPLITS)).float().pow(2).sum().backward()
    grad = adapter.linear_out.weight.grad
    assert torch.count_nonzero(grad[:, :INTER, R:]) == 0 and torch.count_nonzero(grad[:, INTER:, :R]) == 0
    assert torch.count_nonzero(grad[0, :INTER, :R]) > 0 and torch.count_nonzero(grad[0, INTER:, R:]) > 0
    assert torch.count_nonzero(adapter.linear_in.weight.grad) > 0


def test_merge_is_the_gate_and_up_deltas():
    adapter = _fc1()
    _randomize_diagonal(adapter)
    a = adapter.linear_in.weight.float()
    for e in range(E):
        b = adapter.linear_out.weight[e].float()
        merged = LoRAMerge().merge(
            torch.zeros(2 * INTER, H, device="cuda"), b, a, ALPHA, adapter.dim, tp_size=1, tp_group=None
        )
        expected = torch.cat([b[:INTER, :R] @ a[:R], b[INTER:, R:] @ a[R:]]) * (ALPHA / R)
        torch.testing.assert_close(merged, expected)


def test_refuses_fc2_and_expert_tp():
    with pytest.raises(ValueError, match="fc1 only"):
        _fc1(input_is_parallel=True, base_linear_name="decoder.layers.0.mlp.experts.linear_fc2")
    with pytest.raises(ValueError, match="expert_tensor_parallel_size=1"):
        _fc1(config={"expert_tensor_parallel_size": 2})


def test_lora_refuses_the_split_without_shared_outer():
    lora = LoRA(experts_shared_outer_separate_gate_up=True)
    with pytest.raises(ValueError, match="requires experts_shared_outer_loras"):
        lora.transform(nn.Linear(4, 4), name="linear_fc1", prefix="decoder.layers.0.mlp.experts")
