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

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask


class _Bridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):
        return None

    def mapping_registry(self):
        return MegatronMappingRegistry()


class _ExpertSlice:
    """One local expert's slice of a grouped HF tensor (all experts of a layer)."""

    is_grouped_export = True

    def __init__(self, hf_param, expert):
        self.hf_param = hf_param
        self.megatron_param = f"experts.weight{expert}"
        self.expert = expert

    def hf_to_megatron(self, hf_weights, _megatron_module):
        return hf_weights[self.expert].clone()


class _Probe:
    """An ungrouped mapping that records whether the grouped tensor is still alive."""

    def __init__(self, hf_param, alive):
        self.hf_param = hf_param
        self.megatron_param = "dense.weight"
        self.alive = alive

    def hf_to_megatron(self, hf_weights, _megatron_module):
        gc.collect()
        self.alive.append(loaded["grouped"]() is not None)
        return hf_weights


loaded = {}


def _task(mapping, param):
    return WeightConversionTask(
        param_name=mapping.megatron_param,
        global_param_name=mapping.megatron_param,
        mapping=mapping,
        pp_rank=0,
        vp_stage=0,
        megatron_module=Mock(),
        param_weight=param,
    )


class TestGroupedImportCache:
    def test_grouped_tensor_is_loaded_once_and_dropped_after_its_last_task(self, monkeypatch):
        experts = torch.arange(6.0).reshape(2, 3)
        alive = []
        params = [torch.nn.Parameter(torch.zeros(3), requires_grad=False) for _ in range(3)]
        tasks = [
            _task(_ExpertSlice("layer0.experts", 0), params[0]),
            _task(_ExpertSlice("layer0.experts", 1), params[1]),
            _task(_Probe("layer1.dense", alive), params[2]),
        ]
        loads = []

        def load(self, hf_param, state):
            loads.append(hf_param)
            t = state[hf_param].clone()
            if hf_param == "layer0.experts":
                loaded["grouped"] = weakref.ref(t)
            return t

        monkeypatch.setattr(_Bridge, "build_conversion_tasks", lambda self, *_a, **_k: tasks)
        monkeypatch.setattr(_Bridge, "_with_progress_tracking", lambda self, t, *_a, **_k: t)
        monkeypatch.setattr(_Bridge, "_broadcast_shared_embeddings", lambda self, *_a, **_k: None)
        monkeypatch.setattr(_Bridge, "maybe_modify_loaded_hf_weight", load)
        hf = SimpleNamespace(state={"layer0.experts": experts, "layer1.dense": torch.ones(3)}, model_name_or_path="x")

        _Bridge().load_weights_hf_to_megatron(hf, [SimpleNamespace()])

        assert loads == ["layer0.experts", "layer1.dense"]  # the grouped tensor is read once
        assert alive == [False]  # and freed before the next layer's task runs
        torch.testing.assert_close(params[0].detach(), experts[0])
        torch.testing.assert_close(params[1].detach(), experts[1])
