"""ACT-X: ACT with explicit point/task condition token (x).

The only structural change vs ACT:
- An nn.Embedding(num_points, dim_model) maps point ID -> x token
- x token is inserted after z token in the encoder input sequence:
    [z, x, (robot_state), (env_state), (image_feature_map_pixels)]
- encoder_1d_feature_pos_embed has +1 token to accommodate x
- forward() reads point ID from batch["task_index"]
"""

import einops
import torch
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACT, ACTPolicy
from lerobot.policies.act_x.configuration_act_x import ACTXConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACTXPolicy(ACTPolicy):
    """ACT-X Policy: wraps ACTPolicy, replaces self.model with ACTX."""

    config_class = ACTXConfig
    name = "act_x"

    def __init__(self, config: ACTXConfig, **kwargs):
        # Skip ACTPolicy.__init__ to avoid creating ACT model, call grandparent
        super(ACTPolicy, self).__init__(config)
        config.validate_features()
        self.config = config
        self.model = ACTX(config)

        if config.temporal_ensemble_coeff is not None:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )

        self.reset()

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Prepare batch: collect image features into OBS_IMAGES list."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return batch

    def _extract_point_id(self, batch: dict[str, Tensor]) -> Tensor | None:
        """Extract and normalize point_id from batch."""
        point_id = batch.get("task_index", None)
        if point_id is not None:
            point_id = point_id.long()
            if point_id.dim() > 1:
                point_id = point_id.squeeze(-1)
        return point_id

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Forward pass for training. Extracts point_id from batch['task_index']."""
        batch = self._prepare_batch(batch)
        point_id = self._extract_point_id(batch)

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch, point_id=point_id)

        l1_loss = (
            nn.functional.l1_loss(batch[ACTION], actions_hat, reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()))
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + self.config.kl_weight * mean_kld
        else:
            loss = l1_loss

        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Inference: predict action chunk given a point_id."""
        self.eval()
        batch = self._prepare_batch(batch)
        point_id = self._extract_point_id(batch)
        actions, _ = self.model(batch, point_id=point_id)
        return actions

    def set_point_id(self, point_id: int):
        """Set a fixed point_id for inference (from classifier output)."""
        self._inference_point_id = point_id

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for environment interaction, with optional fixed point_id."""
        if hasattr(self, "_inference_point_id"):
            device = batch[OBS_STATE].device if OBS_STATE in batch else next(self.parameters()).device
            batch = dict(batch)
            batch["task_index"] = torch.tensor([self._inference_point_id], device=device)
        return super().select_action(batch)


class ACTX(ACT):
    """ACT with additional point condition token x.

    Encoder input sequence:
        [z_token, x_token, (robot_state), (env_state), (image_feature_map_pixels)]
    """

    def __init__(self, config: ACTXConfig):
        super().__init__(config)

        # Point ID embedding: maps point_id (0~num_points-1) -> dim_model
        self.point_embed = nn.Embedding(config.num_points, config.dim_model)

        # Rebuild encoder_1d_feature_pos_embed with +1 for x token
        n_1d_tokens = 1  # latent z
        n_1d_tokens += 1  # point x
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)

        self._reset_parameters()
        # Re-init point_embed after _reset_parameters (which only touches encoder/decoder)
        nn.init.normal_(self.point_embed.weight, std=0.02)

    def forward(
        self, batch: dict[str, Tensor], point_id: Tensor | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """Forward pass with point condition token x injected after z.

        Follows the exact same pattern as ACT.forward(), with the only addition
        being the x token inserted after z in the encoder input sequence.
        """
        batch_size = (
            batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
        )

        # === VAE encoder (identical to original ACT) ===
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)  # (B, S+2, D)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=vae_encoder_input.device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim],
                dtype=torch.float32,
                device=batch[OBS_STATE].device if self.config.robot_state_feature else batch[OBS_ENV_STATE].device,
            )

        # === Build encoder input tokens (sequence-first: each token is (B, D)) ===
        # Token 1: z (latent)
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        # Token 2: x (point condition) -- THE KEY ADDITION vs ACT
        if point_id is not None:
            x_token = self.point_embed(point_id)  # (B, dim_model)
        else:
            x_token = torch.zeros(batch_size, self.config.dim_model, device=encoder_in_tokens[0].device)
        encoder_in_tokens.append(x_token)

        # Token 3+: robot state
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        # Token 4+: env state
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        # Image tokens
        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # Stack and run encoder
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        # Decoder
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
