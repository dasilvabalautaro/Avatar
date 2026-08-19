from __future__ import annotations

import pytest

from avatar_face.domain.attributes import (
    ATTRIBUTE_CARDINALITIES,
    ATTRIBUTE_ORDER,
    ATTRIBUTE_VOCABULARIES,
    AvatarAttributes,
    CaptionParseError,
    attributes_from_prompt,
    attributes_from_text,
    parse_caption,
)
from avatar_face.domain.models import AvatarPrompt, InvalidPromptError

CAPTION_WITH_ACCESSORY = (
    "flat vector avatar face of an adult, happy expression, square face, porcelain skin tone, "
    "side-parted black hair, green almond eyes with round glasses, sky background"
)
CAPTION_WITHOUT_ACCESSORY = (
    "flat vector avatar face of an adult, calm expression, heart face, deep skin tone, "
    "bob pink hair, gray narrow eyes without accessories, sand background"
)


def test_vocabulary_shape() -> None:
    assert ATTRIBUTE_ORDER == tuple(ATTRIBUTE_VOCABULARIES)
    assert ATTRIBUTE_CARDINALITIES == (4, 4, 6, 4, 6, 4, 3, 6, 5)


def test_parse_caption_roundtrip() -> None:
    for caption in (CAPTION_WITH_ACCESSORY, CAPTION_WITHOUT_ACCESSORY):
        attributes = parse_caption(caption)
        assert attributes.caption() == caption


def test_parse_caption_fields() -> None:
    attributes = parse_caption(CAPTION_WITH_ACCESSORY)
    assert attributes.hair_style == "side-parted"
    assert attributes.hair_color == "black"
    assert attributes.eye_color == "green"
    assert attributes.eye_shape == "almond"
    assert attributes.accessory == "round glasses"


def test_parse_caption_rejects_deviations() -> None:
    with pytest.raises(CaptionParseError):
        parse_caption("avatar face of an adult, happy expression")
    with pytest.raises(CaptionParseError):
        parse_caption(CAPTION_WITH_ACCESSORY.replace("green", "purple"))


def test_attributes_validate_vocabulary() -> None:
    with pytest.raises(ValueError):
        AvatarAttributes(
            expression="angry",
            face_shape="oval",
            skin_tone="light",
            hair_style="short",
            hair_color="brown",
            eye_color="brown",
            eye_shape="almond",
            accessory="none",
            background="sky",
        )


def test_indices_match_vocabulary_order() -> None:
    attributes = parse_caption(CAPTION_WITH_ACCESSORY)
    indices = attributes.indices()
    assert len(indices) == len(ATTRIBUTE_ORDER)
    for name, index in zip(ATTRIBUTE_ORDER, indices, strict=True):
        assert ATTRIBUTE_VOCABULARIES[name][index] == getattr(attributes, name)


def test_free_text_uses_context() -> None:
    attributes = attributes_from_text(
        "confident avatar with curly blue hair, brown round eyes, freckles and mint background"
    )
    assert attributes.expression == "confident"
    assert attributes.hair_style == "curly"
    assert attributes.hair_color == "blue"
    assert attributes.eye_color == "brown"
    assert attributes.eye_shape == "round"
    assert attributes.accessory == "freckles"
    assert attributes.background == "mint"
    # «brown» y «round» sólo asignan con contexto: la piel y la cara quedan neutras.
    assert attributes.skin_tone == "light"
    assert attributes.face_shape == "oval"


def test_free_text_defaults() -> None:
    attributes = attributes_from_text("an adult avatar")
    assert attributes == AvatarAttributes(
        expression="calm",
        face_shape="oval",
        skin_tone="light",
        hair_style="short",
        hair_color="brown",
        eye_color="brown",
        eye_shape="almond",
        accessory="none",
        background="sky",
    )


def test_prompt_flow_keeps_adult_filter() -> None:
    attributes = attributes_from_prompt(AvatarPrompt("smiling adult with sunglasses"))
    assert attributes.accessory == "sunglasses"
    with pytest.raises(InvalidPromptError):
        attributes_from_prompt(AvatarPrompt("smiling boy with sunglasses"))
