"""ACT-X Configuration: ACT with explicit point/task condition token (x)."""

from dataclasses import dataclass, field

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("act_x")
@dataclass
class ACTXConfig(ACTConfig):
    """ACT-X: ACT with an additional condition token x (point ID).

    x is injected as a learnable embedding token into the transformer encoder,
    right after the VAE latent token z:
        [z, x, (robot_state), (env_state), (image_feature_map_pixels)]

    During training, x = ground truth point ID (from task_index).
    During inference, x = classifier prediction.

    Additional args:
        num_points: Number of distinct point/task categories.
        point_embed_dim: Dimension of point embedding (projected to dim_model).
    """
    num_points: int = 10
