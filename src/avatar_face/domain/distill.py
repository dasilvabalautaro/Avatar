from __future__ import annotations

import hashlib
from dataclasses import dataclass

from avatar_face.domain.attributes import ATTRIBUTE_VOCABULARIES, AvatarAttributes

# Mismos strides que el generador procedimental v3 para los atributos que
# aparecen en el caption: la cobertura queda estratificada (diferencia máxima
# de una muestra por categoría) y alineada con la distribución que el maestro
# LoRA vio en entrenamiento.
_STRIDES: dict[str, int] = {
    "expression": 13,
    "face_shape": 23,
    "skin_tone": 5,
    "hair_style": 19,
    "hair_color": 7,
    "eye_color": 11,
    "eye_shape": 37,
    "accessory": 17,
    "background": 3,
}
GENERATOR_ID = "avatarface-distill-captions-v1"


@dataclass(frozen=True, slots=True)
class DistillCaption:
    """Par determinista (caption, seed) que el maestro convertirá en imagen."""

    identifier: str
    caption: str
    attributes: AvatarAttributes
    seed: int
    split: str


def _stratified_choice(values: tuple[str, ...], index: int, seed: int, stride: int) -> str:
    """Idéntico al generador procedimental: cada categoría difiere en ≤1 muestra."""
    size = len(values)
    cycle, position = divmod(index, size)
    return values[(position + cycle * stride + seed) % size]


def _sample_seed(seed: int, index: int) -> int:
    """Seed de 32 bits propia de cada par, independiente del orden de generación."""
    digest = hashlib.sha256(f"{seed}:{index}:distill".encode("ascii")).digest()
    return int.from_bytes(digest[:4], "big")


def build_distill_captions(samples: int, seed: int) -> tuple[DistillCaption, ...]:
    """Lista determinista de pares para el dataset de destilación (ADR 0010)."""
    if not 8 <= samples <= 10_000:
        raise ValueError("El conjunto de destilación debe tener entre 8 y 10,000 pares.")
    pairs = []
    for index in range(samples):
        attributes = AvatarAttributes(
            **{
                name: _stratified_choice(
                    ATTRIBUTE_VOCABULARIES[name], index, seed, _STRIDES[name]
                )
                for name in _STRIDES
            }
        )
        split = "test" if index % 10 == 0 else "validation" if index % 10 == 1 else "train"
        pairs.append(
            DistillCaption(
                identifier=f"avatar-{index:05d}",
                caption=attributes.caption(),
                attributes=attributes,
                seed=_sample_seed(seed, index),
                split=split,
            )
        )
    return tuple(pairs)


def distill_captions_payload(samples: int, seed: int) -> dict[str, object]:
    """Documento serializable que viaja a la instancia GPU (o se regenera allí)."""
    pairs = build_distill_captions(samples, seed)
    return {
        "schema_version": 1,
        "generator": GENERATOR_ID,
        "seed": seed,
        "samples": len(pairs),
        "splits": {
            split: sum(pair.split == split for pair in pairs)
            for split in ("train", "validation", "test")
        },
        "pairs": [
            {
                "identifier": pair.identifier,
                "caption": pair.caption,
                "attributes": {
                    name: getattr(pair.attributes, name) for name in _STRIDES
                },
                "seed": pair.seed,
                "split": pair.split,
            }
            for pair in pairs
        ],
    }
