"""Observation wrapper that adds spaced temporal history for state vectors."""

from collections import deque
from typing import Deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class StateHistoryWrapper(gym.Wrapper):
    """Add `state_history` to dict observations using spaced state taps."""

    def __init__(self, env: gym.Env, history_len: int = 5, history_stride: int = 2):
        super().__init__(env)
        if history_len < 1:
            raise ValueError("history_len must be >= 1")
        if history_stride < 1:
            raise ValueError("history_stride must be >= 1")
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError("StateHistoryWrapper requires Dict observation space")
        if "state" not in env.observation_space.spaces:
            raise KeyError("StateHistoryWrapper requires observation key 'state'")

        self.history_len = int(history_len)
        self.history_stride = int(history_stride)
        self._buffer_size = self.history_len * self.history_stride

        state_space = env.observation_space.spaces["state"]
        if len(state_space.shape) != 1:
            raise ValueError(f"Expected 1D `state`, got shape {state_space.shape}")
        self._state_dim = int(state_space.shape[0])

        spaces_dict = dict(env.observation_space.spaces)
        spaces_dict["state_history"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.history_len, self._state_dim),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(spaces_dict)

        self._state_buffer: Deque[np.ndarray] = deque(maxlen=self._buffer_size)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = np.asarray(obs["state"], dtype=np.float32)
        self._state_buffer.clear()
        for _ in range(self._buffer_size):
            self._state_buffer.append(state.copy())
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = np.asarray(obs["state"], dtype=np.float32)
        self._state_buffer.append(state.copy())
        return self._augment_obs(obs), reward, terminated, truncated, info

    def _augment_obs(self, obs: dict) -> dict:
        if not self._state_buffer:
            state = np.asarray(obs["state"], dtype=np.float32)
            for _ in range(self._buffer_size):
                self._state_buffer.append(state.copy())

        offsets = range((self.history_len - 1) * self.history_stride, -1, -self.history_stride)
        history = np.stack([self._state_buffer[-1 - off] for off in offsets], axis=0).astype(np.float32)

        out = dict(obs)
        out["state_history"] = history
        return out
