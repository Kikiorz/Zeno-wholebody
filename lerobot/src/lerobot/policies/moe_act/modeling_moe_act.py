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
"""Mixture-of-Experts Action Chunking Transformer Policy

Extends the ACT policy by replacing the single FFN in each encoder layer with
N expert FFNs and a learned router (Mixture-of-Experts).
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTEncoderLayer,
    ACTPolicy,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
    get_activation_fn,
)
from lerobot.policies.moe_act.configuration_moe_act import MoEACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class MoEFFN(nn.Module):
    """Single expert feed-forward network module."""

    def __init__(self, dim_model: int, dim_feedforward: int, dropout: float, activation: Callable):
        super().__init__()
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class MoELayer(nn.Module):
    """Mixture-of-Experts layer: Router + N Expert FFNs."""

    def __init__(
        self,
        dim_model: int,
        dim_feedforward: int,
        num_experts: int,
        num_experts_per_token: int,
        dropout: float,
        activation: Callable,
    ):
        super().__init__()
        self.router = nn.Linear(dim_model, num_experts)
        self.experts = nn.ModuleList([
            MoEFFN(dim_model, dim_feedforward, dropout, activation)
            for _ in range(num_experts)
        ])
        self.num_experts_per_token = num_experts_per_token
        self.balance_loss = torch.tensor(0.0)

    def forward(self, x: Tensor) -> Tensor:
        # x: (seq_len, batch, dim_model) → flatten for routing
        orig_shape = x.shape
        flat_x = x.reshape(-1, x.shape[-1])  # (S*B, D)

        # Router
        logits = self.router(flat_x)  # (S*B, num_experts)
        gates = F.softmax(logits, dim=-1)  # (S*B, num_experts)

        # Top-k selection
        top_vals, top_idx = gates.topk(self.num_experts_per_token, dim=-1)  # (S*B, k)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)  # renormalize

        # Compute expert outputs and weighted sum
        output = torch.zeros_like(flat_x)  # (S*B, D)
        for k_i in range(self.num_experts_per_token):
            expert_indices = top_idx[:, k_i]  # (S*B,)
            expert_gates = top_vals[:, k_i]  # (S*B,)
            for e_idx in range(len(self.experts)):
                mask = (expert_indices == e_idx)
                if mask.any():
                    expert_input = flat_x[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += expert_gates[mask].unsqueeze(-1) * expert_output

        # Compute load balancing loss
        # f_i = fraction of tokens routed to expert i
        # P_i = mean gate probability for expert i
        N = len(self.experts)
        f = torch.zeros(N, device=x.device)
        for e_idx in range(N):
            f[e_idx] = (top_idx == e_idx).float().sum() / (top_idx.shape[0] * self.num_experts_per_token)
        P = gates.mean(dim=0)  # (num_experts,)
        self.balance_loss = N * (f * P).sum()

        return output.reshape(orig_shape)


class MoEACTEncoderLayer(nn.Module):
    """ACT encoder layer with MoE replacing the single FFN."""

    def __init__(self, config: MoEACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # MoE replaces the single FFN
        self.moe = MoELayer(
            config.dim_model,
            config.dim_feedforward,
            config.num_experts,
            config.num_experts_per_token,
            config.dropout,
            get_activation_fn(config.feedforward_activation),
        )

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        # MoE-FFN replaces the original single FFN
        x = self.moe(x)

        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class MoEACTEncoder(nn.Module):
    """Encoder that uses MoEACTEncoderLayer for the main encoder and standard ACTEncoderLayer for VAE encoder."""

    def __init__(self, config: MoEACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        if is_vae_encoder:
            # VAE encoder doesn't need MoE — uses standard ACTEncoderLayer
            self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        else:
            # Main encoder uses MoE
            self.layers = nn.ModuleList([MoEACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class MoEACT(nn.Module):
    """Action Chunking Transformer with Mixture-of-Experts in the encoder.

    Identical to ACT except:
    - The main encoder uses MoEACTEncoder (with MoE layers)
    - The VAE encoder stays as standard (no MoE)
    - Provides get_moe_balance_loss() to collect balance losses from all MoE layers
    """

    def __init__(self, config: MoEACTConfig):
        super().__init__()
        self.config = config

        # BERT style VAE encoder (standard, no MoE)
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction (identical to ACT)
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Main encoder uses MoE
        self.encoder = MoEACTEncoder(config)
        # Decoder stays shared (identical to ACT)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections (identical to ACT)
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_moe_balance_loss(self) -> Tensor:
        """Collect and sum balance losses from all MoE layers in the encoder."""
        total_balance_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.encoder.layers:
            if isinstance(layer, MoEACTEncoderLayer):
                total_balance_loss = total_balance_loss + layer.moe.balance_loss
        return total_balance_loss

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the MoE Action Chunking Transformer (with optional VAE encoder).

        Identical to ACT.forward() — the MoE is internal to the encoder layers.
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()

            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class MoEACTPolicy(ACTPolicy):
    """MoE-ACT Policy: ACT with Mixture-of-Experts in the encoder layers.

    Overrides:
    - Uses MoEACT model instead of ACT
    - Adds MoE load balancing loss to the training objective
    """

    config_class = MoEACTConfig
    name = "moe_act"

    def __init__(
        self,
        config: MoEACTConfig,
        **kwargs,
    ):
        # Call PreTrainedPolicy.__init__ directly to avoid ACTPolicy creating an ACT model
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        self.model = MoEACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation.

        Adds MoE load balancing loss to the standard ACT loss.
        """
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        loss = l1_loss

        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = loss + mean_kld * self.config.kl_weight

        # MoE load balancing loss
        balance_loss = self.model.get_moe_balance_loss()
        loss_dict["balance_loss"] = balance_loss.item()
        loss = loss + self.config.moe_balance_weight * balance_loss

        return loss, loss_dict
