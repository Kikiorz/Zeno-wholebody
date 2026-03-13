#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("moe_act")
@dataclass
class MoEACTConfig(ACTConfig):
    """Configuration class for the Mixture-of-Experts Action Chunking Transformers policy.

    Extends ACTConfig with MoE-specific parameters. The MoE replaces the single FFN in each
    encoder layer with N expert FFNs and a learned router.

    Args:
        num_experts: Number of expert FFNs per encoder layer.
        num_experts_per_token: Top-k routing (number of experts selected per token).
        moe_balance_weight: Weight for the load balancing loss term.
    """

    # MoE parameters
    num_experts: int = 10
    num_experts_per_token: int = 2
    moe_balance_weight: float = 0.01
