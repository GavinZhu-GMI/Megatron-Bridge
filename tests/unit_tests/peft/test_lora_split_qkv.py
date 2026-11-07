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

"""LoRA(split_qkv_adapters=True) builds linear_qkv's adapters exactly as CanonicalLoRA does."""

from unittest.mock import patch

import torch.nn as nn

from megatron.bridge.peft.canonical_lora import CanonicalLoRA, LoRALinearSplitQKV
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.lora_layers import LoRALinear
from megatron.bridge.peft.utils import AdapterAttributes


HIDDEN, KV_CHANNELS, HEADS, GROUPS, RANK = 64, 8, 8, 2, 4


class _Config:
    kv_channels = KV_CHANNELS
    num_attention_heads = HEADS
    num_query_groups = GROUPS
    sequence_parallel = False
    moe_router_topk = None


class _MegatronLinear(nn.Module):
    """Stands in for a Megatron parallel linear (not nn.Linear, so the parallel-adapter path runs)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.config = _Config()

    def forward(self, x):
        return x, None


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_qkv = _MegatronLinear(HIDDEN, KV_CHANNELS * (HEADS + 2 * GROUPS))
        self.linear_proj = _MegatronLinear(KV_CHANNELS * HEADS, HIDDEN)


def _attrs(module, is_expert=False):
    return AdapterAttributes(
        input_is_parallel=False,
        in_features=module.in_features,
        out_features=module.out_features,
        disable_tensor_parallel_comm=False,
        disable_sequence_parallel_comm=True,
        base_linear_is_parallel=True,
    )


def _transform(peft, module_path: str):
    """Apply the PEFT with the attribute lookup and ParallelLinearAdapter stubbed; return the model and calls."""
    calls = []

    def adapter(in_features, out_features, dim=None, **kwargs):
        calls.append((in_features, out_features, dim if dim is not None else kwargs["dim"]))
        return nn.Linear(1, 1)

    with (
        patch(f"megatron.bridge.peft.{module_path}.get_adapter_attributes_from_linear", side_effect=_attrs),
        patch(f"megatron.bridge.peft.{module_path}.ParallelLinearAdapter", side_effect=adapter),
    ):
        model = peft(_Attention(), training=True)
    return model, calls


def test_split_qkv_matches_canonical_lora():
    ours, our_calls = _transform(LoRA(target_modules=["linear_qkv"], dim=RANK, split_qkv_adapters=True), "lora")
    canon, canon_calls = _transform(
        CanonicalLoRA(target_modules=["linear_q", "linear_k", "linear_v"], dim=RANK), "canonical_lora"
    )
    assert isinstance(ours.linear_qkv, LoRALinearSplitQKV)
    assert isinstance(canon.linear_qkv, LoRALinearSplitQKV)
    # q, k, v in that order: (in, out, rank) each
    expected = [
        (HIDDEN, KV_CHANNELS * HEADS, RANK),
        (HIDDEN, KV_CHANNELS * GROUPS, RANK),
        (HIDDEN, KV_CHANNELS * GROUPS, RANK),
    ]
    assert our_calls == canon_calls == expected


def test_split_qkv_leaves_other_targets_fused():
    model, calls = _transform(
        LoRA(target_modules=["linear_qkv", "linear_proj"], dim=RANK, split_qkv_adapters=True), "lora"
    )
    assert isinstance(model.linear_qkv, LoRALinearSplitQKV)
    assert isinstance(model.linear_proj, LoRALinear)
    assert calls[-1] == (KV_CHANNELS * HEADS, HIDDEN, RANK)


def test_default_keeps_one_fused_qkv_adapter():
    model, calls = _transform(LoRA(target_modules=["linear_qkv"], dim=RANK), "lora")
    assert isinstance(model.linear_qkv, LoRALinear)
    assert calls == [(HIDDEN, KV_CHANNELS * (HEADS + 2 * GROUPS), RANK)]
