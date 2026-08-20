from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from avatar_face.domain.attributes import (  # noqa: E402
    ATTRIBUTE_VOCABULARIES,
    AvatarAttributes,
    attributes_from_text,
)
from avatar_face.infrastructure.rendering.avatar_renderer import (  # noqa: E402
    FlatVectorAvatarRenderer,
)
from avatar_face.presentation.cli import main  # noqa: E402

BASE = AvatarAttributes(
    expression="smiling",
    face_shape="oval",
    skin_tone="light",
    hair_style="short",
    hair_color="brown",
    eye_color="brown",
    eye_shape="almond",
    accessory="none",
    background="sky",
)


def test_render_size_and_mode() -> None:
    for size in (256, 512):
        image = FlatVectorAvatarRenderer(image_size=size).render(BASE)
        assert image.size == (size, size)
        assert image.mode == "RGB"


def test_render_is_deterministic() -> None:
    renderer = FlatVectorAvatarRenderer()
    assert renderer.render(BASE).tobytes() == renderer.render(BASE).tobytes()


def test_every_vocabulary_value_renders() -> None:
    """Ningún valor del vocabulario cerrado puede romper el dibujo."""
    renderer = FlatVectorAvatarRenderer(image_size=64)
    for name, values in ATTRIBUTE_VOCABULARIES.items():
        for value in values:
            attributes = AvatarAttributes(**{**vars_of(BASE), name: value})
            image = renderer.render(attributes)
            assert image.size == (64, 64)


def vars_of(attributes: AvatarAttributes) -> dict[str, str]:
    return {name: getattr(attributes, name) for name in ATTRIBUTE_VOCABULARIES}


def test_attributes_change_the_pixels() -> None:
    """Atributos distintos deben producir imágenes distintas, no un dibujo fijo."""
    renderer = FlatVectorAvatarRenderer(image_size=64)
    reference = renderer.render(BASE).tobytes()
    for name, values in ATTRIBUTE_VOCABULARIES.items():
        other = next(v for v in values if v != getattr(BASE, name))
        changed = renderer.render(AvatarAttributes(**{**vars_of(BASE), name: other}))
        assert changed.tobytes() != reference, f"{name}={other} no cambió la imagen"


def test_render_command_writes_image(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "avatar.png"
    command = ["render", "adult with curly blue hair and sunglasses", "--output", str(output)]
    assert main(command) == 0
    assert output.is_file()
    payload = json.loads(capsys.readouterr().out)
    assert payload["attributes"]["hair_color"] == "blue"
    assert payload["attributes"]["accessory"] == "sunglasses"
    assert "of an adult" in payload["caption"]


def test_render_command_rejects_minors(tmp_path: Path) -> None:
    """RF-09: el filtro de sólo adultos actúa antes de dibujar."""
    output = tmp_path / "no.png"
    assert main(["render", "a little boy with blue hair", "--output", str(output)]) == 2
    assert not output.exists()


def test_free_text_reaches_the_renderer() -> None:
    attributes = attributes_from_text("confident adult, square face, deep skin, sand background")
    image = FlatVectorAvatarRenderer(image_size=64).render(attributes)
    assert image.size == (64, 64)
