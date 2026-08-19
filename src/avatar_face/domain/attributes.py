from __future__ import annotations

import re
from dataclasses import dataclass

from avatar_face.domain.models import AvatarPrompt

# Vocabulario cerrado del generador procedimental v3 (sólo los atributos que
# aparecen en el caption; brow_style y nose_style no se describen). El orden de
# cada tupla es canónico: define el índice de embedding del estudiante y no
# debe reordenarse sin crear una versión nueva del vocabulario.
ATTRIBUTE_VOCABULARIES: dict[str, tuple[str, ...]] = {
    "expression": ("smiling", "calm", "happy", "confident"),
    "face_shape": ("round", "oval", "square", "heart"),
    "skin_tone": ("porcelain", "light", "golden", "tan", "brown", "deep"),
    "hair_style": ("short", "curly", "side-parted", "bob"),
    "hair_color": ("black", "brown", "auburn", "blonde", "blue", "pink"),
    "eye_color": ("brown", "blue", "green", "gray"),
    "eye_shape": ("almond", "round", "narrow"),
    "accessory": (
        "none",
        "round glasses",
        "square glasses",
        "sunglasses",
        "earrings",
        "freckles",
    ),
    "background": ("coral", "mint", "sky", "lavender", "sand"),
}
ATTRIBUTE_ORDER: tuple[str, ...] = tuple(ATTRIBUTE_VOCABULARIES)
ATTRIBUTE_CARDINALITIES: tuple[int, ...] = tuple(
    len(ATTRIBUTE_VOCABULARIES[name]) for name in ATTRIBUTE_ORDER
)

# Valores neutros para prompts libres que no mencionan un atributo.
DEFAULT_ATTRIBUTES: dict[str, str] = {
    "expression": "calm",
    "face_shape": "oval",
    "skin_tone": "light",
    "hair_style": "short",
    "hair_color": "brown",
    "eye_color": "brown",
    "eye_shape": "almond",
    "accessory": "none",
    "background": "sky",
}

_CAPTION_PREFIX = "flat vector avatar face of an adult, "

_FACE_PATTERN = re.compile(r"\b(round|oval|square|heart)\s+face\b")
_SKIN_PATTERN = re.compile(r"\b(porcelain|light|golden|tan|brown|deep)\s+skin\b")
_HAIR_PATTERN = re.compile(
    r"\b(?:(short|curly|side-parted|bob)\s+)?(black|brown|auburn|blonde|blue|pink)\s+hair\b"
)
_HAIR_STYLE_PATTERN = re.compile(r"\b(short|curly|side-parted|bob)\s+hair\b")
_EYES_PATTERN = re.compile(
    r"\b(?:(brown|blue|green|gray)\s+)?(?:(almond|round|narrow)\s+)?eyes\b"
)
_EXPRESSION_PATTERN = re.compile(r"\b(smiling|calm|happy|confident)\b")
_BACKGROUND_PATTERN = re.compile(r"\b(coral|mint|sky|lavender|sand)\s+background\b")


class CaptionParseError(ValueError):
    """El caption no sigue la plantilla del generador procedimental v3."""


@dataclass(frozen=True, slots=True)
class AvatarAttributes:
    """Atributos estructurados de un rostro; condicionan al estudiante (ADR 0010)."""

    expression: str
    face_shape: str
    skin_tone: str
    hair_style: str
    hair_color: str
    eye_color: str
    eye_shape: str
    accessory: str
    background: str

    def __post_init__(self) -> None:
        for name in ATTRIBUTE_ORDER:
            value = getattr(self, name)
            if value not in ATTRIBUTE_VOCABULARIES[name]:
                raise ValueError(f"Valor de {name} fuera del vocabulario: {value!r}.")

    def indices(self) -> tuple[int, ...]:
        """Índices de embedding en el orden canónico de ATTRIBUTE_ORDER."""
        return tuple(
            ATTRIBUTE_VOCABULARIES[name].index(getattr(self, name)) for name in ATTRIBUTE_ORDER
        )

    def caption(self) -> str:
        """Caption con la plantilla exacta del generador procedimental v3."""
        accessory = (
            " without accessories" if self.accessory == "none" else f" with {self.accessory}"
        )
        return (
            f"flat vector avatar face of an adult, {self.expression} expression, "
            f"{self.face_shape} face, {self.skin_tone} skin tone, "
            f"{self.hair_style} {self.hair_color} hair, {self.eye_color} "
            f"{self.eye_shape} eyes{accessory}, {self.background} background"
        )


def _strip_suffix(segment: str, suffix: str) -> str:
    if not segment.endswith(suffix):
        raise CaptionParseError(f"Segmento inesperado: {segment!r} (se esperaba …{suffix!r}).")
    return segment[: -len(suffix)]


def parse_caption(caption: str) -> AvatarAttributes:
    """Parser estricto de la plantilla del generador; falla ante cualquier desvío."""
    if not caption.startswith(_CAPTION_PREFIX):
        raise CaptionParseError("El caption no empieza con la plantilla del generador v3.")
    segments = caption[len(_CAPTION_PREFIX) :].split(", ")
    if len(segments) != 6:
        raise CaptionParseError(f"El caption tiene {len(segments)} segmentos; se esperaban 6.")
    hair_words = _strip_suffix(segments[3], " hair").split(" ")
    if len(hair_words) != 2:
        raise CaptionParseError(f"Segmento de pelo inválido: {segments[3]!r}.")
    eyes_segment = segments[4]
    if eyes_segment.endswith(" eyes without accessories"):
        eyes_words = eyes_segment.removesuffix(" eyes without accessories").split(" ")
        accessory = "none"
    elif " eyes with " in eyes_segment:
        eyes_part, accessory = eyes_segment.split(" eyes with ", 1)
        eyes_words = eyes_part.split(" ")
    else:
        raise CaptionParseError(f"Segmento de ojos inválido: {eyes_segment!r}.")
    if len(eyes_words) != 2:
        raise CaptionParseError(f"Segmento de ojos inválido: {eyes_segment!r}.")
    try:
        return AvatarAttributes(
            expression=_strip_suffix(segments[0], " expression"),
            face_shape=_strip_suffix(segments[1], " face"),
            skin_tone=_strip_suffix(segments[2], " skin tone"),
            hair_style=hair_words[0],
            hair_color=hair_words[1],
            eye_color=eyes_words[0],
            eye_shape=eyes_words[1],
            accessory=accessory,
            background=_strip_suffix(segments[5], " background"),
        )
    except ValueError as error:
        raise CaptionParseError(str(error)) from error


def attributes_from_text(text: str) -> AvatarAttributes:
    """Parser tolerante para prompts libres; usa valores neutros para lo ausente.

    Los atributos ambiguos («brown», «round», «light») se resuelven por
    contexto («brown hair», «round face»); un término suelto sin contexto no
    asigna nada. La validación de sólo-adultos (RF-09) ocurre antes, en
    `AvatarPrompt`; usar `attributes_from_prompt` para el flujo del producto.
    """
    normalized = text.lower()
    values = dict(DEFAULT_ATTRIBUTES)
    for accessory in ATTRIBUTE_VOCABULARIES["accessory"][1:]:
        if accessory in normalized:
            values["accessory"] = accessory
            break
    if match := _EXPRESSION_PATTERN.search(normalized):
        values["expression"] = match.group(1)
    if match := _FACE_PATTERN.search(normalized):
        values["face_shape"] = match.group(1)
    if match := _SKIN_PATTERN.search(normalized):
        values["skin_tone"] = match.group(1)
    if match := _HAIR_PATTERN.search(normalized):
        if match.group(1):
            values["hair_style"] = match.group(1)
        values["hair_color"] = match.group(2)
    if match := _HAIR_STYLE_PATTERN.search(normalized):
        values["hair_style"] = match.group(1)
    if match := _EYES_PATTERN.search(normalized):
        if match.group(1):
            values["eye_color"] = match.group(1)
        if match.group(2):
            values["eye_shape"] = match.group(2)
    if match := _BACKGROUND_PATTERN.search(normalized):
        values["background"] = match.group(1)
    return AvatarAttributes(**values)


def attributes_from_prompt(prompt: AvatarPrompt) -> AvatarAttributes:
    """Flujo del producto: el prompt ya pasó el contrato y el filtro RF-09."""
    return attributes_from_text(prompt.text)
