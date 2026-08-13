import pytest

from avatar_face.domain.models import AndroidDevice, AvatarPrompt, InvalidPromptError


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


def test_android_device_is_ready_only_in_device_state() -> None:
    assert AndroidDevice("serial", "device").ready
    assert not AndroidDevice("serial", "unauthorized").ready
