"""Temporal-Progress Router model.

Structurally identical to ACT (non-VAE path) but outputs (B, 100, 1) expert
routing logits instead of (B, 100, 17) actions.  Visual encoder loaded from
stageA checkpoint, frozen backbone option via separate param groups.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import einops
import torch
import torch.nn.functional as F
import torchvision
from safetensors.torch import load_file as load_safetensors
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lerobot" / "src"))

from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
)


@dataclass
class RouterConfig:
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    pre_norm: bool = False
    dropout: float = 0.01
    chunk_size: int = 100
    latent_dim: int = 32


class _RouterConfigAdapter:
    """Thin adapter so ACTEncoder / ACTDecoder can consume a RouterConfig
    through the same attribute interface they expect from ACTConfig."""

    def __init__(self, cfg: RouterConfig):
        self.dim_model = cfg.dim_model
        self.n_heads = cfg.n_heads
        self.dim_feedforward = cfg.dim_feedforward
        self.feedforward_activation = cfg.feedforward_activation
        self.n_encoder_layers = cfg.n_encoder_layers
        self.n_decoder_layers = cfg.n_decoder_layers
        self.pre_norm = cfg.pre_norm
        self.dropout = cfg.dropout
        self.chunk_size = cfg.chunk_size
        self.latent_dim = cfg.latent_dim
        # ACTEncoder checks this when is_vae_encoder=False
        self.n_vae_encoder_layers = cfg.n_encoder_layers


class TemporalProgressRouter(nn.Module):
    def __init__(self, config: RouterConfig | None = None):
        super().__init__()
        if config is None:
            config = RouterConfig()
        self.config = config
        act_cfg = _RouterConfigAdapter(config)

        # ── Backbone (ResNet34 + FrozenBatchNorm2d) ──
        backbone_model = torchvision.models.resnet34(
            replace_stride_with_dilation=[False, False, False],
            weights=None,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(
            backbone_model, return_layers={"layer4": "feature_map"}
        )

        # ── Encoder input projections ──
        self.encoder_img_feat_input_proj = nn.Conv2d(512, config.dim_model, kernel_size=1)
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
        self.encoder_robot_state_input_proj = nn.Linear(17, config.dim_model)
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        # 2 one-dimensional tokens: latent + robot_state
        self.encoder_1d_feature_pos_embed = nn.Embedding(2, config.dim_model)

        # ── Transformer encoder & decoder ──
        self.encoder = ACTEncoder(act_cfg)
        self.decoder = ACTDecoder(act_cfg)
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # ── Router head (NEW — replaces action_head) ──
        self.router_head = nn.Linear(config.dim_model, 1)

    def forward(self, images: list[Tensor], state: Tensor) -> Tensor:
        """
        Args:
            images: list of 3 camera tensors, each (B, 3, 224, 224)
            state:  (B, 17) normalized robot state
        Returns:
            logits: (B, 100) raw routing logits  (P(expert_B) before sigmoid)
        """
        batch_size = state.shape[0]
        device = state.device

        # Latent = zeros (no VAE)
        latent_sample = torch.zeros(
            batch_size, self.config.latent_dim, dtype=torch.float32, device=device
        )

        # Build encoder input tokens and positional embeddings
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        # Robot state token
        encoder_in_tokens.append(self.encoder_robot_state_input_proj(state))

        # Image tokens (3 cameras)
        for img in images:
            cam_features = self.backbone(img)["feature_map"]
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(
                dtype=cam_features.dtype
            )
            cam_features = self.encoder_img_feat_input_proj(cam_features)
            cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
            cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
            encoder_in_tokens.extend(list(cam_features))
            encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)

        # Transformer forward
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            self.config.chunk_size, batch_size, self.config.dim_model,
            dtype=encoder_in_pos_embed.dtype, device=device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)  # (B, chunk_size, dim_model)

        logits = self.router_head(decoder_out).squeeze(-1)  # (B, 100)
        return logits

    # ── Weight loading ──────────────────────────────────────────────────────

    @classmethod
    def load_from_act_checkpoint(cls, ckpt_dir: str | Path, config: RouterConfig | None = None):
        """Load ACT weights from a stageA checkpoint, skipping VAE and action_head keys."""
        ckpt_dir = Path(ckpt_dir)
        safetensors_path = ckpt_dir / "model.safetensors"

        model = cls(config)

        state_dict = load_safetensors(str(safetensors_path))

        # Strip 'model.' prefix and filter out vae_encoder* / action_head*
        filtered = {}
        for k, v in state_dict.items():
            if not k.startswith("model."):
                continue
            short_key = k[len("model."):]
            if short_key.startswith("vae_encoder"):
                continue
            if short_key.startswith("action_head"):
                continue
            filtered[short_key] = v

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        # Expected missing: router_head.weight, router_head.bias
        print(f"[Router] Loaded ACT weights from {safetensors_path}")
        print(f"  Missing keys  (expected — new head): {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        return model

    # ── Param groups ────────────────────────────────────────────────────────

    def get_param_groups(self, lr_backbone: float, lr_rest: float):
        """Return optimizer param groups with separate LR for backbone."""
        backbone_params = list(self.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        rest_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": rest_params, "lr": lr_rest},
        ]
