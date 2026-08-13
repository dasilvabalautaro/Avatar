from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeasibilityProfile:
    """Dimensiones fijas para probar el presupuesto del modelo móvil."""

    name: str
    vocabulary_size: int
    maximum_tokens: int
    text_width: int
    model_width: int
    depth: int
    attention_heads: int
    latent_size: int
    latent_channels: int
    image_size: int
    steps: int
    decoder_channels: tuple[int, ...]
    decoder_scales: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.model_width % self.attention_heads:
            raise ValueError("model_width debe ser divisible por attention_heads.")
        if self.image_size != self.latent_size * math.prod(self.effective_decoder_scales):
            raise ValueError("decoder_channels no produce el image_size solicitado.")
        if self.steps < 1:
            raise ValueError("steps debe ser positivo.")

    @property
    def latent_tokens(self) -> int:
        return self.latent_size**2

    @property
    def effective_decoder_scales(self) -> tuple[int, ...]:
        if self.decoder_scales:
            if len(self.decoder_scales) != len(self.decoder_channels) - 1:
                raise ValueError("decoder_scales debe describir cada transición.")
            return self.decoder_scales
        return (2,) * (len(self.decoder_channels) - 1)

    @property
    def fp32_parameter_budget_bytes(self) -> int:
        return self.estimated_parameters * 4

    @property
    def int8_parameter_budget_bytes(self) -> int:
        return self.estimated_parameters

    @property
    def estimated_parameters(self) -> int:
        """Estimación conservadora para ordenar perfiles sin importar PyTorch."""
        text = self.vocabulary_size * self.text_width
        conditioning = self.text_width * self.model_width + self.steps * self.model_width
        latent_projection = self.latent_channels * self.model_width
        transformer = self.depth * (12 * self.model_width**2 + 13 * self.model_width)
        output = self.model_width * self.latent_channels
        decoder = self.latent_channels * self.decoder_channels[0] * 9
        for input_channels, output_channels in zip(
            self.decoder_channels, self.decoder_channels[1:], strict=False
        ):
            decoder += input_channels * output_channels * 9
        decoder += self.decoder_channels[-1] * 3 * 9
        return text + conditioning + latent_projection + transformer + output + decoder


FEASIBILITY_PROFILES: dict[str, FeasibilityProfile] = {
    "micro": FeasibilityProfile(
        name="micro",
        vocabulary_size=2_048,
        maximum_tokens=16,
        text_width=64,
        model_width=64,
        depth=2,
        attention_heads=4,
        latent_size=8,
        latent_channels=4,
        image_size=256,
        steps=1,
        decoder_channels=(64, 64, 32, 16, 8, 8),
    ),
    "bridge": FeasibilityProfile(
        name="bridge",
        vocabulary_size=4_096,
        maximum_tokens=16,
        text_width=128,
        model_width=320,
        depth=6,
        attention_heads=8,
        latent_size=8,
        latent_channels=4,
        image_size=256,
        steps=2,
        decoder_channels=(128, 128, 64, 32, 16, 8),
    ),
    "bridge-slim": FeasibilityProfile(
        name="bridge-slim",
        vocabulary_size=4_096,
        maximum_tokens=16,
        text_width=128,
        model_width=320,
        depth=6,
        attention_heads=8,
        latent_size=8,
        latent_channels=4,
        image_size=256,
        steps=2,
        decoder_channels=(64, 64, 32, 16, 8, 4),
    ),
    "bridge-fast": FeasibilityProfile(
        name="bridge-fast",
        vocabulary_size=4_096,
        maximum_tokens=16,
        text_width=128,
        model_width=320,
        depth=6,
        attention_heads=8,
        latent_size=8,
        latent_channels=4,
        image_size=256,
        steps=2,
        decoder_channels=(32, 16, 8, 4),
        decoder_scales=(4, 4, 2),
    ),
    "target": FeasibilityProfile(
        name="target",
        vocabulary_size=8_192,
        maximum_tokens=32,
        text_width=256,
        model_width=832,
        depth=16,
        attention_heads=8,
        latent_size=8,
        latent_channels=4,
        image_size=256,
        steps=4,
        decoder_channels=(256, 256, 128, 64, 32, 16),
    ),
    "stress": FeasibilityProfile(
        name="stress",
        vocabulary_size=16_384,
        maximum_tokens=32,
        text_width=384,
        model_width=1_024,
        depth=20,
        attention_heads=16,
        latent_size=8,
        latent_channels=4,
        image_size=256,
        steps=4,
        decoder_channels=(384, 384, 192, 96, 48, 24),
    ),
}


def get_feasibility_profile(name: str) -> FeasibilityProfile:
    try:
        return FEASIBILITY_PROFILES[name]
    except KeyError as error:
        available = ", ".join(sorted(FEASIBILITY_PROFILES))
        raise ValueError(f"Perfil desconocido: {name}. Disponibles: {available}.") from error
