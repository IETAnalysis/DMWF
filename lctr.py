from __future__ import annotations

import torch
import torch.nn as nn


class LengthCoupledTraceRepresentation(nn.Module):
    """Construct the six-channel LCTR representation from observable metadata."""

    def __init__(self, clip_value: float = 0.0, epsilon: float = 0.0):
        super().__init__()
        if clip_value <= 0 or epsilon <= 0:
            raise ValueError("clip_value and epsilon must be positive")
        self.clip_value = float(clip_value)
        self.epsilon = float(epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 3:
            raise ValueError("input must have shape [batch, 3, sequence_length]")
        direction = x[:, 0:1]
        iat = torch.clamp(x[:, 1:2], min=0.0, max=self.clip_value)
        length = torch.clamp(x[:, 2:3], min=0.0, max=self.clip_value)
        signed_length = direction * length
        length_to_iat = torch.clamp(
            length / (iat + self.epsilon), min=0.0, max=self.clip_value
        )
        previous = torch.cat([torch.zeros_like(length[..., :1]), length[..., :-1]], dim=-1)
        adjacent_difference = torch.clamp(
            length - previous, min=-self.clip_value, max=self.clip_value
        )
        return torch.cat(
            [
                direction,
                iat,
                length,
                signed_length,
                length_to_iat,
                adjacent_difference,
            ],
            dim=1,
        )
