from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lctr import LengthCoupledTraceRepresentation


def slot_diversity_loss(slot_features: torch.Tensor) -> torch.Tensor:
    if slot_features.size(1) <= 1:
        return slot_features.new_tensor(0.0)
    normalized = F.normalize(slot_features, dim=-1)
    similarity = torch.matmul(normalized, normalized.transpose(1, 2)).abs()
    diagonal = torch.eye(
        similarity.size(1), device=similarity.device, dtype=torch.bool
    ).unsqueeze(0)
    return similarity.masked_select(~diagonal).mean()


class PositionalEncoding(nn.Module):
    def __init__(
        self, dimension: int, dropout: float, max_length: int, encoding_base: float
    ):
        super().__init__()
        encoding = torch.zeros(max_length, dimension)
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(encoding_base) / dimension)
        )
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("encoding", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.encoding[:, : x.size(1)])


class SEBlock1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 0):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale(x)


class ResidualSEBlock1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 0,
        se_reduction: int = 0,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.transform = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_channels, output_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(output_channels),
        )
        self.shortcut = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)
        self.se = SEBlock1d(output_channels, se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.activation(self.transform(x) + self.shortcut(x)))


class LocalEvidenceBranch(nn.Module):
    def __init__(
        self,
        input_channels: int = 0,
        embedding_dimension: int = 0,
        first_channels: int = 0,
        second_channels: int = 0,
        kernel_size: int = 0,
        se_reduction: int = 0,
        pool_kernel: int = 0,
        pool_stride: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.network = nn.Sequential(
            ResidualSEBlock1d(
                input_channels, first_channels, kernel_size, se_reduction
            ),
            nn.MaxPool1d(pool_kernel, stride=pool_stride),
            nn.Dropout(dropout),
            ResidualSEBlock1d(
                first_channels, second_channels, kernel_size, se_reduction
            ),
            nn.MaxPool1d(pool_kernel, stride=pool_stride),
            nn.Dropout(dropout),
            ResidualSEBlock1d(
                second_channels, embedding_dimension, kernel_size, se_reduction
            ),
            nn.MaxPool1d(pool_kernel, stride=pool_stride),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).transpose(1, 2)


class MultiSlotEvidenceDecoder(nn.Module):
    def __init__(
        self,
        embedding_dimension: int = 0,
        slots: int = 0,
        attention_heads: int = 0,
        layers: int = 0,
        dropout: float = 0.0,
        feedforward_multiplier: int = 0,
        slot_query_scale: float = 0.0,
    ):
        super().__init__()
        self.slot_queries = nn.Parameter(
            torch.randn(1, slots, embedding_dimension) * slot_query_scale
        )
        layer = nn.TransformerDecoderLayer(
            d_model=embedding_dimension,
            nhead=attention_heads,
            dim_feedforward=embedding_dimension * feedforward_multiplier,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.normalization = nn.LayerNorm(embedding_dimension)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        queries = self.slot_queries.expand(memory.size(0), -1, -1)
        return self.normalization(self.decoder(queries, memory))


class DMWFModel(nn.Module):
    """DMWF with LCTR, twin residual-SE branches, and multi-slot decoding."""

    def __init__(
        self,
        num_classes: int,
        max_cardinality: int = 0,
        sequence_length: int = 0,
        max_slots: int = 0,
        embedding_dimension: int = 0,
        attention_heads: int = 0,
        encoder_layers: int = 0,
        decoder_layers: int = 0,
        dropout: float = 0.0,
        auxiliary_weight: float = 0.0,
        statistics_weight: float = 0.0,
        lctr_clip_value: float = 0.0,
        lctr_epsilon: float = 0.0,
        first_branch_channels: int = 0,
        second_branch_channels: int = 0,
        convolution_kernel_size: int = 0,
        se_reduction: int = 0,
        pool_kernel: int = 0,
        pool_stride: int = 0,
        feedforward_multiplier: int = 0,
        statistics_pool_bins: int = 0,
        statistics_hidden_dimension: int = 0,
        positional_encoding_base: float = 0.0,
        positional_max_length: int = 0,
        slot_query_scale: float = 0.0,
        noisy_or_epsilon: float = 0.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.max_cardinality = int(max_cardinality)
        self.sequence_length = int(sequence_length)
        self.max_slots = int(max_slots)
        self.auxiliary_weight = float(auxiliary_weight)
        self.statistics_weight = float(statistics_weight)
        self.noisy_or_epsilon = float(noisy_or_epsilon)
        self.lctr = LengthCoupledTraceRepresentation(lctr_clip_value, lctr_epsilon)

        branch_arguments = dict(
            input_channels=3,
            embedding_dimension=embedding_dimension,
            first_channels=first_branch_channels,
            second_channels=second_branch_channels,
            kernel_size=convolution_kernel_size,
            se_reduction=se_reduction,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            dropout=dropout,
        )
        self.direct_branch = LocalEvidenceBranch(**branch_arguments)
        self.coupled_branch = LocalEvidenceBranch(**branch_arguments)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dimension * 2, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position = PositionalEncoding(
            embedding_dimension,
            dropout,
            max_length=positional_max_length,
            encoding_base=positional_encoding_base,
        )
        encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=attention_heads,
            dim_feedforward=embedding_dimension * feedforward_multiplier,
            dropout=dropout,
            batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder, num_layers=encoder_layers)
        self.slot_decoder = MultiSlotEvidenceDecoder(
            embedding_dimension,
            max_slots,
            attention_heads,
            decoder_layers,
            dropout,
            feedforward_multiplier,
            slot_query_scale,
        )
        self.slot_classifier = nn.Linear(embedding_dimension, num_classes)
        self.auxiliary_classifier = nn.Linear(embedding_dimension * 2, num_classes)
        self.statistics_branch = nn.Sequential(
            nn.AdaptiveAvgPool1d(statistics_pool_bins),
            nn.Flatten(),
            nn.Linear(6 * statistics_pool_bins, statistics_hidden_dimension),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(statistics_hidden_dimension),
            nn.Dropout(dropout),
            nn.Linear(statistics_hidden_dimension, num_classes),
        )
        self.cardinality_head = nn.Linear(
            embedding_dimension * 3, max_cardinality + 1
        )

    def noisy_or_logits(self, slot_logits: torch.Tensor) -> torch.Tensor:
        epsilon = self.noisy_or_epsilon
        slot_probability = torch.sigmoid(slot_logits).clamp(epsilon, 1.0 - epsilon)
        probability = 1.0 - torch.prod(1.0 - slot_probability, dim=1)
        return torch.logit(probability.clamp(epsilon, 1.0 - epsilon))

    def build_lctr(self, x: torch.Tensor) -> torch.Tensor:
        return self.lctr(x)

    def forward(self, x: torch.Tensor):
        representation = self.build_lctr(x)
        direct = self.direct_branch(representation[:, :3])
        coupled = self.coupled_branch(representation[:, 3:])
        memory = self.context_encoder(
            self.position(self.fusion(torch.cat([direct, coupled], dim=-1)))
        )
        maximum = torch.max(memory, dim=1).values
        mean = torch.mean(memory, dim=1)
        global_context = torch.cat([maximum, mean], dim=-1)
        slots = self.slot_decoder(memory)
        slot_logits = self.slot_classifier(slots)
        auxiliary_logits = self.auxiliary_classifier(global_context)
        statistics_logits = self.statistics_branch(representation)
        classification_logits = (
            self.noisy_or_logits(slot_logits)
            + self.auxiliary_weight * auxiliary_logits
            + self.statistics_weight * statistics_logits
        )
        cardinality_logits = self.cardinality_head(
            torch.cat([global_context, torch.mean(slots, dim=1)], dim=-1)
        )
        return (
            classification_logits,
            cardinality_logits,
            auxiliary_logits,
            statistics_logits,
            slot_logits,
            slots,
        )


# Historical name retained for loading code that imported the development class.
DWMFModelV3 = DMWFModel
