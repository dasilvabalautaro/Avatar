from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from avatar_face.domain.feasibility import FeasibilityProfile


class FeasibilityAttention(nn.Module):
    """Atención explícita compuesta por operadores ONNX comunes."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.scale = 1.0 / math.sqrt(self.head_width)
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)

    def forward(self, sequence: Tensor) -> Tensor:
        batch, tokens, width = sequence.shape
        qkv = self.qkv(sequence).reshape(batch, tokens, 3, self.heads, self.head_width)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        weights = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) * self.scale, -1)
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, tokens, width)
        return cast(Tensor, self.output(attended))


class FeasibilityBlock(nn.Module):
    """Bloque DiT pequeño y exportable, sin kernels personalizados."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = FeasibilityAttention(width, heads)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.SiLU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        sequence = sequence + self.attention(self.attention_norm(sequence))
        return cast(Tensor, sequence + self.mlp(self.mlp_norm(sequence)))


class FeasibilityDecoder(nn.Module):
    """Decoder RGB sencillo basado en resize y convoluciones."""

    def __init__(
        self,
        latent_channels: int,
        channels: tuple[int, ...],
        scales: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.input = nn.Conv2d(latent_channels, channels[0], 3, padding=1)
        stages: list[nn.Module] = []
        for input_channels, output_channels, scale in zip(
            channels[:-1], channels[1:], scales, strict=True
        ):
            stages.extend(
                [
                    nn.Upsample(scale_factor=float(scale), mode="nearest"),
                    nn.Conv2d(input_channels, output_channels, 3, padding=1),
                    nn.SiLU(),
                ]
            )
        self.stages = nn.Sequential(*stages)
        self.output = nn.Conv2d(channels[-1], 3, 3, padding=1)

    def forward(self, latent: Tensor) -> Tensor:
        return torch.tanh(self.output(self.stages(self.input(latent))))


class AvatarFaceFeasibilityModel(nn.Module):
    """Grafo sintético representativo; no genera avatares de calidad."""

    def __init__(self, profile: FeasibilityProfile) -> None:
        super().__init__()
        self.profile = profile
        self.token_embedding = nn.Embedding(profile.vocabulary_size, profile.text_width)
        self.conditioning = nn.Linear(profile.text_width, profile.model_width)
        self.step_embedding = nn.Embedding(profile.steps, profile.model_width)
        self.latent_input = nn.Conv2d(profile.latent_channels, profile.model_width, 1)
        self.position = nn.Parameter(torch.zeros(1, profile.latent_tokens, profile.model_width))
        self.blocks = nn.ModuleList(
            FeasibilityBlock(profile.model_width, profile.attention_heads)
            for _ in range(profile.depth)
        )
        self.output_norm = nn.LayerNorm(profile.model_width)
        self.latent_output = nn.Linear(profile.model_width, profile.latent_channels)
        self.decoder = FeasibilityDecoder(
            profile.latent_channels,
            profile.decoder_channels,
            profile.effective_decoder_scales,
        )

    def forward(self, token_ids: Tensor, latent: Tensor) -> Tensor:
        batch = latent.shape[0]
        condition = self.conditioning(self.token_embedding(token_ids).mean(dim=1))
        sequence = self.latent_input(latent).flatten(2).transpose(1, 2)
        sequence = sequence + self.position + condition.unsqueeze(1)
        for step in range(self.profile.steps):
            step_ids = torch.full((batch,), step, dtype=torch.long, device=latent.device)
            sequence = sequence + self.step_embedding(step_ids).unsqueeze(1)
            for block in self.blocks:
                sequence = block(sequence)
        predicted = self.latent_output(self.output_norm(sequence))
        predicted = predicted.transpose(1, 2).reshape(
            batch,
            self.profile.latent_channels,
            self.profile.latent_size,
            self.profile.latent_size,
        )
        return cast(Tensor, self.decoder(predicted))


class FeasibilityTextEncoder(nn.Module):
    """Vista exportable del encoder que conserva los pesos del modelo completo."""

    def __init__(self, model: AvatarFaceFeasibilityModel) -> None:
        super().__init__()
        self.token_embedding = model.token_embedding
        self.conditioning = model.conditioning

    def forward(self, token_ids: Tensor) -> Tensor:
        return cast(Tensor, self.conditioning(self.token_embedding(token_ids).mean(dim=1)))


class FeasibilityDenoiser(nn.Module):
    """Vista exportable del denoiser sin incluir encoder ni decoder."""

    def __init__(self, model: AvatarFaceFeasibilityModel) -> None:
        super().__init__()
        self.profile = model.profile
        self.step_embedding = model.step_embedding
        self.latent_input = model.latent_input
        self.position = model.position
        self.blocks = model.blocks
        self.output_norm = model.output_norm
        self.latent_output = model.latent_output

    def forward(self, condition: Tensor, latent: Tensor) -> Tensor:
        batch = latent.shape[0]
        sequence = self.latent_input(latent).flatten(2).transpose(1, 2)
        sequence = sequence + self.position + condition.unsqueeze(1)
        for step in range(self.profile.steps):
            step_ids = torch.full((batch,), step, dtype=torch.long, device=latent.device)
            sequence = sequence + self.step_embedding(step_ids).unsqueeze(1)
            for block in self.blocks:
                sequence = block(sequence)
        predicted = self.latent_output(self.output_norm(sequence))
        return cast(
            Tensor,
            predicted.transpose(1, 2).reshape(
                batch,
                self.profile.latent_channels,
                self.profile.latent_size,
                self.profile.latent_size,
            ),
        )


def component_models(
    model: AvatarFaceFeasibilityModel,
) -> dict[str, tuple[nn.Module, tuple[Tensor, ...], tuple[str, ...], str]]:
    """Construye vistas con pesos compartidos y contratos ONNX explícitos."""
    token_ids, latent = example_inputs(model.profile)
    condition = torch.zeros((1, model.profile.model_width), dtype=torch.float32)
    return {
        "encoder": (
            FeasibilityTextEncoder(model).eval(),
            (token_ids,),
            ("token_ids",),
            "condition",
        ),
        "denoiser": (
            FeasibilityDenoiser(model).eval(),
            (condition, latent),
            ("condition", "latent"),
            "predicted_latent",
        ),
        "decoder": (
            model.decoder.eval(),
            (latent,),
            ("latent",),
            "image",
        ),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def example_inputs(profile: FeasibilityProfile) -> tuple[Tensor, Tensor]:
    token_ids = torch.zeros((1, profile.maximum_tokens), dtype=torch.long)
    latent = torch.zeros(
        (1, profile.latent_channels, profile.latent_size, profile.latent_size),
        dtype=torch.float32,
    )
    return token_ids, latent
