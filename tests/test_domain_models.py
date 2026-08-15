import pytest

from avatar_face.domain.models import (
    AndroidDevice,
    AvatarPrompt,
    InvalidPromptError,
)


def test_avatar_prompt_normalizes_text() -> None:
    prompt = AvatarPrompt("  avatar con cabello azul  ", seed=7, image_size=256)

    assert prompt.text == "avatar con cabello azul"
    assert prompt.seed == 7


@pytest.mark.parametrize(
    ("text", "seed", "image_size"),
    [
        (" ", 42, 256),
        ("avatar", -1, 256),
        ("avatar", 2**32, 256),
        ("avatar", 42, 128),
    ],
)
def test_avatar_prompt_rejects_invalid_values(text: str, seed: int, image_size: int) -> None:
    with pytest.raises(InvalidPromptError):
        AvatarPrompt(text, seed, image_size)


@pytest.mark.parametrize(
    "text",
    [
        "avatar de un niño con gorra",
        "retrato de una nina sin acentos",
        "cute kid avatar with freckles",
        "Teenager Avatar With Blue Hair",
        "avatar of a 3 year old toddler",
        "avatar de una persona de 17 años",
    ],
)
def test_avatar_prompt_rejects_minor_references(text: str) -> None:
    """RF-09: el producto sólo genera rostros de adultos."""
    with pytest.raises(InvalidPromptError, match="menor de edad"):
        AvatarPrompt(text, seed=42, image_size=256)


@pytest.mark.parametrize(
    "text",
    [
        "avatar de una persona de 18 años",
        "portrait of a 45 year old woman",
        "avatar con sonrisa juvenil",
        "flat vector avatar face of an adult",
    ],
)
def test_avatar_prompt_accepts_adult_references(text: str) -> None:
    prompt = AvatarPrompt(text, seed=42, image_size=256)

    assert prompt.text == text.strip()


def test_android_device_is_ready_only_in_device_state() -> None:
    assert AndroidDevice("serial", "device").ready
    assert not AndroidDevice("serial", "unauthorized").ready
