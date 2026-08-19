from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import Tensor, nn

from avatar_face.domain.attributes import ATTRIBUTE_CARDINALITIES


@dataclass(frozen=True, slots=True)
class StudentUNetConfig:
    """Presupuesto arquitectónico del estudiante (ADR 0010): 30–60 M parámetros.

    El mismo grafo sirve a las dos formulaciones: en la directa el ratio es
    constante y el modelo aprende (ruido, atributos) → imagen; en la de
    difusión predice epsilon. Todos los operadores son estándar y exportables
    a ONNX (conv, group norm, atención explícita), como en el spike sintético.
    """

    image_size: int = 256
    base_channels: int = 96
    channel_multipliers: tuple[int, ...] = (1, 2, 3, 4)
    residual_blocks_per_level: int = 2
    attention_resolutions: tuple[int, ...] = (32,)
    attention_heads: int = 4
    condition_dim: int = 512
    attribute_embedding_dim: int = 64
    attribute_cardinalities: tuple[int, ...] = field(default=ATTRIBUTE_CARDINALITIES)

    def __post_init__(self) -> None:
        if self.image_size % (2 ** (len(self.channel_multipliers) - 1)):
            raise ValueError("image_size debe ser divisible por el factor total de submuestreo.")
        if not self.attribute_cardinalities:
            raise ValueError("Se requiere al menos un atributo de condicionamiento.")


def _timestep_embedding(ratio: Tensor, dim: int) -> Tensor:
    """Embedding sinusoidal del ratio del scheduler (0–1), escalado a 0–1000."""
    half = dim // 2
    frequencies = torch.exp(
        torch.arange(half, dtype=torch.float32, device=ratio.device)
        * (-math.log(10_000.0) / (half - 1))
    )
    angles = ratio.float().mul(1000.0)[:, None] * frequencies[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=1).to(ratio.dtype)


class ResidualBlock(nn.Module):
    """Bloque residual con modulación FiLM desde la condición (tiempo + atributos)."""

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.film = nn.Linear(condition_dim, out_channels * 2)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        hidden = self.conv1(nn.functional.silu(self.norm1(x)))
        scale, shift = self.film(condition)[:, :, None, None].chunk(2, dim=1)
        hidden = nn.functional.silu(self.norm2(hidden) * (1 + scale) + shift)
        return cast(Tensor, self.skip(x) + self.conv2(hidden))


class SelfAttention2d(nn.Module):
    """Atención explícita sobre el mapa espacial, compatible con exportación ONNX."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("Los canales de atención deben ser divisibles por las cabezas.")
        self.heads = heads
        self.scale = 1.0 / math.sqrt(channels // heads)
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        query, key, value = (
            self.qkv(self.norm(x))
            .reshape(batch, 3, self.heads, channels // self.heads, height * width)
            .unbind(dim=1)
        )
        weights = torch.softmax(query.transpose(-2, -1) @ key * self.scale, dim=-1)
        attended = (value @ weights.transpose(-2, -1)).reshape(batch, channels, height, width)
        return cast(Tensor, x + self.out(attended))


class StudentUNet(nn.Module):
    """U-Net condicional a 256 px que sustituye todo el stack Würstchen (ADR 0010)."""

    def __init__(self, config: StudentUNetConfig) -> None:
        super().__init__()
        self.config = config
        channels = [config.base_channels * m for m in config.channel_multipliers]
        condition_dim = config.condition_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(config.base_channels, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.attribute_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, config.attribute_embedding_dim)
            for cardinality in config.attribute_cardinalities
        )
        self.attribute_mlp = nn.Sequential(
            nn.Linear(
                config.attribute_embedding_dim * len(config.attribute_cardinalities),
                condition_dim,
            ),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        self.stem = nn.Conv2d(3, channels[0], 3, padding=1)
        resolutions = [config.image_size // (2**level) for level in range(len(channels))]

        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels: list[int] = [channels[0]]
        current = channels[0]
        for level, target in enumerate(channels):
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(config.residual_blocks_per_level):
                blocks.append(ResidualBlock(current, target, condition_dim))
                attentions.append(
                    SelfAttention2d(target, config.attention_heads)
                    if resolutions[level] in config.attention_resolutions
                    else nn.Identity()
                )
                current = target
                skip_channels.append(current)
            self.down_blocks.append(blocks)
            self.down_attentions.append(attentions)
            if level + 1 < len(channels):
                self.downsamples.append(nn.Conv2d(current, current, 3, stride=2, padding=1))
                skip_channels.append(current)
            else:
                self.downsamples.append(nn.Identity())

        self.middle_block1 = ResidualBlock(current, current, condition_dim)
        self.middle_attention = SelfAttention2d(current, config.attention_heads)
        self.middle_block2 = ResidualBlock(current, current, condition_dim)

        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level in reversed(range(len(channels))):
            target = channels[level]
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(config.residual_blocks_per_level + 1):
                blocks.append(
                    ResidualBlock(current + skip_channels.pop(), target, condition_dim)
                )
                attentions.append(
                    SelfAttention2d(target, config.attention_heads)
                    if resolutions[level] in config.attention_resolutions
                    else nn.Identity()
                )
                current = target
            self.up_blocks.append(blocks)
            self.up_attentions.append(attentions)
            self.upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2.0, mode="nearest"),
                    nn.Conv2d(current, current, 3, padding=1),
                )
                if level > 0
                else nn.Identity()
            )

        self.head = nn.Sequential(
            nn.GroupNorm(32, current),
            nn.SiLU(),
            nn.Conv2d(current, 3, 3, padding=1),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, x: Tensor, ratio: Tensor, attributes: Tensor) -> Tensor:
        """x: (b, 3, s, s); ratio: (b,) en [0, 1]; attributes: (b, n) índices."""
        condition = self.time_mlp(
            _timestep_embedding(ratio, self.config.base_channels)
        ) + self.attribute_mlp(
            torch.cat(
                [
                    embedding(attributes[:, index])
                    for index, embedding in enumerate(self.attribute_embeddings)
                ],
                dim=1,
            )
        )

        hidden = self.stem(x)
        skips = [hidden]
        for blocks, attentions, downsample in zip(
            self.down_blocks, self.down_attentions, self.downsamples, strict=True
        ):
            for block, attention in zip(blocks, attentions, strict=True):
                assert isinstance(block, ResidualBlock)
                hidden = attention(block(hidden, condition))
                skips.append(hidden)
            if not isinstance(downsample, nn.Identity):
                hidden = downsample(hidden)
                skips.append(hidden)

        hidden = self.middle_block2(
            self.middle_attention(self.middle_block1(hidden, condition)), condition
        )

        for blocks, attentions, upsample in zip(
            self.up_blocks, self.up_attentions, self.upsamples, strict=True
        ):
            for block, attention in zip(blocks, attentions, strict=True):
                assert isinstance(block, ResidualBlock)
                hidden = attention(block(torch.cat([hidden, skips.pop()], dim=1), condition))
            hidden = upsample(hidden)

        return cast(Tensor, self.head(hidden))
