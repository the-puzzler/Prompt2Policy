"""Small feature extractor with temporal state processing."""

from typing import Dict

import torch
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TinyCombinedExtractor(BaseFeaturesExtractor):
    """Extract compact features from image + state history."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        image_feature_dim: int = 24,
        state_hidden_dim: int = 32,
    ):
        if "state_history" in observation_space.spaces:
            state_history_shape = observation_space.spaces["state_history"].shape
            self.history_len = int(state_history_shape[0])
            self.state_dim = int(state_history_shape[1])
        else:
            self.history_len = 1
            self.state_dim = int(observation_space.spaces["state"].shape[0])

        super().__init__(observation_space, features_dim=state_hidden_dim + image_feature_dim)

        image_space = observation_space.spaces["image"]
        n_input_channels = image_space.shape[0]

        self.image_extractor = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, image_feature_dim),
            nn.ReLU(),
        )

        self.state_projector = nn.Sequential(
            nn.Linear(self.state_dim, state_hidden_dim),
            nn.ReLU(),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(state_hidden_dim, state_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(state_hidden_dim, state_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = observations["image"].float() / 255.0
        image_features = self.image_extractor(image)

        if "state_history" in observations:
            state_history = observations["state_history"].float()
        else:
            state_history = observations["state"].float().unsqueeze(1)
        projected = self.state_projector(state_history)
        temporal_in = projected.transpose(1, 2)
        temporal_features = self.temporal_conv(temporal_in)
        temporal_features = self.temporal_pool(temporal_features).squeeze(-1)

        return torch.cat([temporal_features, image_features], dim=1)
