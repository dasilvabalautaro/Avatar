from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image, ImageDraw

from avatar_face.domain.attributes import AvatarAttributes

# Paleta del estilo «vector plano», tomada como referencia visual de las
# salidas del maestro (docs/experiments/student-lite-2026-08-20.md).
BACKGROUNDS: dict[str, str] = {
    "coral": "#F98F7F",
    "mint": "#7ED9B4",
    "sky": "#5EC8F2",
    "lavender": "#B9A6E8",
    "sand": "#F0DCC0",
}
SKIN_TONES: dict[str, tuple[str, str]] = {
    "porcelain": ("#F7DCC8", "#E8C4AC"),
    "light": ("#EFC7A6", "#DDAE89"),
    "golden": ("#D9A06B", "#C28755"),
    "tan": ("#C08454", "#A76C41"),
    "brown": ("#96603C", "#7D4E2F"),
    "deep": ("#66402C", "#523222"),
}
HAIR_COLORS: dict[str, tuple[str, str]] = {
    "black": ("#1D1A21", "#332E38"),
    "brown": ("#5A3A26", "#714C34"),
    "auburn": ("#8E3F26", "#A85434"),
    "blonde": ("#E0BA6C", "#EFD293"),
    "blue": ("#2F4FA3", "#4468BF"),
    "pink": ("#DE5FA6", "#EC7FBB"),
}
EYE_COLORS: dict[str, str] = {
    "brown": "#5B3720",
    "blue": "#2E6FA8",
    "green": "#3F7A4E",
    "gray": "#5E6B75",
}
SHIRT_COLORS: tuple[str, ...] = ("#F2F2F0", "#E7746A", "#4E7FBF", "#5FAE84", "#E9B34C")
LINE = "#2A2229"
SCLERA = "#FFFDFA"
MOUTH = "#C4544F"
FRECKLE = "#9C5F3E"
GOLD = "#F2C14E"

# El dibujo se hace a este múltiplo del tamaño pedido y se reduce con Lanczos:
# los bordes quedan suavizados sin perder el aspecto de vector plano, que es
# justo lo que fallaba en las salidas difusas del estudiante neuronal.
SUPERSAMPLE = 4

BoxFn = Callable[..., tuple[int, ...]]
WidthFn = Callable[[float], int]


def _draw_head(
    draw: ImageDraw.ImageDraw, box: BoxFn, skin: str, attributes: AvatarAttributes
) -> None:
    radius = {"round": 74, "oval": 64, "square": 38, "heart": 66}[attributes.face_shape]
    top, bottom = (34, 220) if attributes.face_shape == "oval" else (38, 216)
    draw.rounded_rectangle(box(64, top, 192, bottom), radius=box(radius)[0], fill=skin)
    if attributes.face_shape == "heart":
        draw.polygon(box(92, 186, 128, 226, 164, 186), fill=skin)


def _draw_hair_back(
    draw: ImageDraw.ImageDraw, box: BoxFn, hair: str, attributes: AvatarAttributes
) -> None:
    """Melena por detrás de la cabeza; sólo la tienen los estilos largos."""
    if attributes.hair_style == "bob":
        draw.rounded_rectangle(box(50, 44, 206, 202), radius=box(62)[0], fill=hair)
    elif attributes.hair_style == "curly":
        draw.ellipse(box(48, 38, 208, 174), fill=hair)


def _draw_eyes(
    draw: ImageDraw.ImageDraw, box: BoxFn, width: WidthFn, attributes: AvatarAttributes
) -> None:
    iris = EYE_COLORS[attributes.eye_color]
    if attributes.eye_shape == "almond":
        half_width, half_height, radius = 18, 11, 8
    elif attributes.eye_shape == "narrow":
        half_width, half_height, radius = 18, 8, 8
    else:
        half_width, half_height, radius = 15, 14, 9
    for center in (103, 153):
        draw.ellipse(
            box(center - half_width, 133 - half_height, center + half_width, 133 + half_height),
            fill=SCLERA,
        )
        draw.ellipse(box(center - radius, 133 - radius, center + radius, 133 + radius), fill=iris)
        draw.ellipse(box(center - 4, 129, center + 4, 137), fill=LINE)
        draw.ellipse(box(center + 1, 127, center + 6, 132), fill=SCLERA)
    for left, right in ((85, 120), (136, 171)):
        draw.line(
            box(left, 113, (left + right) / 2, 110, right, 112),
            fill=LINE,
            width=width(4),
            joint="curve",
        )


def _draw_mouth(
    draw: ImageDraw.ImageDraw, box: BoxFn, width: WidthFn, attributes: AvatarAttributes
) -> None:
    if attributes.expression in {"smiling", "happy"}:
        draw.chord(box(104, 160, 152, 196), 0, 180, fill=MOUTH)
        draw.chord(box(104, 160, 152, 177), 0, 180, fill=SCLERA)
    elif attributes.expression == "confident":
        draw.line(
            box(106, 178, 128, 185, 150, 175), fill=MOUTH, width=width(6), joint="curve"
        )
    else:
        draw.ellipse(box(113, 172, 143, 187), fill=MOUTH)


def _draw_hair_sides(draw: ImageDraw.ImageDraw, box: BoxFn, hair: str, bottom: float) -> None:
    """Mechones en las sienes: sin ellos el `chord` superior parece un gorro."""
    draw.rounded_rectangle(box(58, 70, 84, bottom), radius=box(13)[0], fill=hair)
    draw.rounded_rectangle(box(172, 70, 198, bottom), radius=box(13)[0], fill=hair)


def _draw_hair_front(
    draw: ImageDraw.ImageDraw,
    box: BoxFn,
    hair: str,
    hair_light: str,
    attributes: AvatarAttributes,
) -> None:
    style = attributes.hair_style
    if style == "short":
        _draw_hair_sides(draw, box, hair, 124)
        draw.chord(box(58, 28, 198, 168), 180, 360, fill=hair)
        draw.chord(box(74, 46, 182, 148), 180, 360, fill=hair_light)
    elif style == "bob":
        _draw_hair_sides(draw, box, hair, 186)
        draw.chord(box(56, 32, 200, 152), 180, 360, fill=hair)
    elif style == "curly":
        _draw_hair_sides(draw, box, hair, 132)
        for x in range(64, 202, 24):
            draw.ellipse(box(x - 21, 30, x + 21, 88), fill=hair)
        draw.chord(box(56, 32, 200, 152), 180, 360, fill=hair)
    else:  # side-parted
        _draw_hair_sides(draw, box, hair, 134)
        draw.chord(box(58, 28, 198, 164), 180, 360, fill=hair)
        draw.polygon(box(98, 36, 194, 58, 190, 98, 148, 60), fill=hair_light)


def _draw_accessory(
    draw: ImageDraw.ImageDraw, box: BoxFn, width: WidthFn, attributes: AvatarAttributes
) -> None:
    accessory = attributes.accessory
    if accessory in {"round glasses", "square glasses", "sunglasses"}:
        left = box(79, 112, 127, 154)
        right = box(129, 112, 177, 154)
        if accessory == "round glasses":
            draw.ellipse(left, outline=LINE, width=width(4))
            draw.ellipse(right, outline=LINE, width=width(4))
        elif accessory == "square glasses":
            draw.rounded_rectangle(left, radius=box(9)[0], outline=LINE, width=width(4))
            draw.rounded_rectangle(right, radius=box(9)[0], outline=LINE, width=width(4))
        else:
            draw.rounded_rectangle(left, radius=box(11)[0], fill=LINE)
            draw.rounded_rectangle(right, radius=box(11)[0], fill=LINE)
        draw.line(box(123, 130, 133, 130), fill=LINE, width=width(4))
        draw.line(box(60, 124, 79, 132), fill=LINE, width=width(4))
        draw.line(box(177, 132, 196, 124), fill=LINE, width=width(4))
    elif accessory == "earrings":
        draw.ellipse(box(62, 154, 78, 172), fill=GOLD)
        draw.ellipse(box(178, 154, 194, 172), fill=GOLD)
    elif accessory == "freckles":
        for x, y in ((99, 152), (109, 159), (147, 159), (157, 152), (104, 166), (152, 166)):
            draw.ellipse(box(x - 3, y - 3, x + 3, y + 3), fill=FRECKLE)


@dataclass(frozen=True, slots=True)
class FlatVectorAvatarRenderer:
    """Dibuja el avatar desde los atributos, sin modelo neuronal.

    Las coordenadas del diseño están sobre un lienzo de 256 px y se escalan al
    tamaño pedido, igual que en el generador procedimental del dataset.
    """

    image_size: int = 256

    def render(self, attributes: AvatarAttributes) -> Image.Image:
        factor = self.image_size * SUPERSAMPLE / 256
        canvas = round(256 * factor)

        def box(*values: float) -> tuple[int, ...]:
            return tuple(round(value * factor) for value in values)

        def width(value: float) -> int:
            return max(1, round(value * factor))

        skin, skin_shadow = SKIN_TONES[attributes.skin_tone]
        hair, hair_light = HAIR_COLORS[attributes.hair_color]
        shirt = SHIRT_COLORS[
            (len(attributes.expression) + len(attributes.background)) % len(SHIRT_COLORS)
        ]
        image = Image.new("RGB", (canvas, canvas), BACKGROUNDS[attributes.background])
        draw = ImageDraw.Draw(image)

        # Torso y cuello: anclan el rostro y evitan el efecto «cabeza flotante».
        draw.ellipse(box(26, 218, 230, 334), fill=shirt)
        draw.rounded_rectangle(box(107, 172, 149, 228), radius=box(20)[0], fill=skin_shadow)
        draw.ellipse(box(58, 118, 82, 158), fill=skin)
        draw.ellipse(box(174, 118, 198, 158), fill=skin)

        _draw_hair_back(draw, box, hair, attributes)
        _draw_head(draw, box, skin, attributes)
        # Sombra lateral: da volumen sin romper el plano de color.
        draw.chord(box(66, 44, 190, 214), 64, 116, fill=skin_shadow)
        _draw_eyes(draw, box, width, attributes)
        # Nariz: sólo una sombra corta, como en la referencia.
        draw.arc(box(120, 144, 136, 166), 210, 330, fill=skin_shadow, width=width(4))
        _draw_mouth(draw, box, width, attributes)
        _draw_hair_front(draw, box, hair, hair_light, attributes)
        _draw_accessory(draw, box, width, attributes)

        return image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
