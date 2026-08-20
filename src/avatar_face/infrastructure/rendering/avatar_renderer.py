from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PIL import Image, ImageDraw

from avatar_face.domain.attributes import AvatarAttributes
from avatar_face.infrastructure.rendering.geometry import (
    Point,
    catmull_rom,
    ellipse_points,
    mirror,
)
from avatar_face.infrastructure.rendering.palette import (
    BACKGROUNDS,
    CLOTHING_COLORS,
    EYE_COLORS,
    GOLD,
    HAIR_COLORS,
    LINE,
    LIP,
    SCLERA,
    SKIN_TONES,
    TEETH,
    Rgb,
    hex_to_rgb,
    mix,
    shade,
)

# El dibujo se hace a este múltiplo del tamaño pedido y se reduce con Lanczos:
# los bordes quedan suavizados sin perder el plano de color.
SUPERSAMPLE = 4

# Anclajes de los rasgos sobre el lienzo de referencia de 256 px. Se mantienen
# fijos aunque cambie la silueta, para que la cara no se descomponga.
EYE_Y = 139.0
EYE_DX = 26.0
BROW_Y = 116.0
NOSE_TOP = 150.0
NOSE_BOTTOM = 170.0
MOUTH_Y = 186.0
CENTER = 128.0

Painter = Callable[[Sequence[Point], Rgb], None]
LinePainter = Callable[[Sequence[Point], Rgb, float], None]
BoxFn = Callable[[float, float, float, float], list[float]]


@dataclass(frozen=True, slots=True)
class FaceShape:
    """Medias anchuras de la silueta a cada altura, más el alto del rostro."""

    temple: float
    cheek: float
    jaw: float
    chin: float
    chin_y: float
    top: float


FACE_SHAPES: dict[str, FaceShape] = {
    "oval": FaceShape(temple=57, cheek=63, jaw=51, chin=25, chin_y=216, top=50),
    "round": FaceShape(temple=61, cheek=67, jaw=61, chin=33, chin_y=207, top=54),
    "square": FaceShape(temple=62, cheek=65, jaw=63, chin=41, chin_y=209, top=52),
    "heart": FaceShape(temple=63, cheek=65, jaw=45, chin=19, chin_y=214, top=52),
    "long": FaceShape(temple=54, cheek=59, jaw=49, chin=25, chin_y=226, top=44),
    "diamond": FaceShape(temple=51, cheek=68, jaw=47, chin=23, chin_y=216, top=54),
}


def face_outline(shape: FaceShape) -> list[Point]:
    """Contorno cerrado y suave del rostro, simétrico respecto al eje central."""
    right: list[Point] = [
        (CENTER, shape.top),
        (CENTER + shape.temple * 0.72, shape.top + 12),
        (CENTER + shape.temple, shape.top + 44),
        (CENTER + shape.cheek, 132.0),
        (CENTER + shape.jaw, 176.0),
        (CENTER + shape.chin, shape.chin_y - 12),
        (CENTER, shape.chin_y),
    ]
    return catmull_rom([*right, *mirror(right)[1:-1]], samples=10)


def _hair_cap(
    shape: FaceShape, volume: float, side_y: float, hairline: float, peak: float = 0.0
) -> list[Point]:
    """Casquete de pelo: sube por las sienes, corona la cabeza y baja a la frente.

    El contorno se recorre en un único sentido —lateral izquierdo hacia arriba,
    coronilla, lateral derecho hacia abajo y línea del pelo de vuelta— porque
    un polígono que salta de un lado a otro se cierra sobre sí mismo y abre una
    muesca en la coronilla.
    """
    width = shape.temple + 5
    # La coronilla es un arco elíptico real: un polígono de puntos sueltos
    # producía una silueta de caja en vez de pelo.
    crown = ellipse_points(
        (CENTER, shape.top + 30), width, 30 + volume + peak, 180, 360, steps=18
    )
    points: list[Point] = [
        (CENTER - width + 7, side_y),
        (CENTER - width - 1, side_y - 28),
        *crown,
        (CENTER + width + 1, side_y - 28),
        (CENTER + width - 7, side_y),
        (CENTER + width - 6, hairline + 12),
        (CENTER + width * 0.62, hairline),
        (CENTER, hairline - 7),
        (CENTER - width * 0.62, hairline),
        (CENTER - width + 6, hairline + 12),
    ]
    return catmull_rom(points, samples=8)


def _draw_hair_back(
    fill: Painter, shape: FaceShape, attributes: AvatarAttributes, hair: Rgb
) -> None:
    """Masa de pelo por detrás de la cabeza; sólo la tienen los estilos largos."""
    style = attributes.hair_style
    dark = shade(hair, -0.18)
    right: list[Point]
    if style == "long":
        right = [
            (CENTER + 4, shape.top - 6),
            (CENTER + shape.temple + 18, shape.top + 40),
            (CENTER + shape.cheek + 22, 150.0),
            (CENTER + shape.cheek + 16, 232.0),
            (CENTER + 40, 244.0),
            (CENTER, 240.0),
        ]
        fill(catmull_rom([*right, *mirror(right)[1:-1]], samples=10), dark)
    elif style == "bob":
        right = [
            (CENTER + 4, shape.top - 4),
            (CENTER + shape.temple + 14, shape.top + 40),
            (CENTER + shape.cheek + 16, 148.0),
            (CENTER + shape.jaw + 16, 196.0),
            (CENTER + 30, 206.0),
            (CENTER, 202.0),
        ]
        fill(catmull_rom([*right, *mirror(right)[1:-1]], samples=10), dark)
    elif style == "afro":
        fill(
            ellipse_points(
                (CENTER, shape.top + 36), shape.temple + 27, shape.temple + 30, 0, 360, steps=30
            ),
            dark,
        )
    elif style == "ponytail":
        right = [
            (CENTER + shape.temple, 96.0),
            (CENTER + shape.temple + 34, 118.0),
            (CENTER + shape.temple + 40, 178.0),
            (CENTER + shape.temple + 18, 206.0),
            (CENTER + shape.temple + 4, 168.0),
        ]
        fill(catmull_rom(right, samples=10), dark)
    elif style == "bun":
        fill(
            catmull_rom(
                [
                    (CENTER - 27, shape.top - 18),
                    (CENTER, shape.top - 40),
                    (CENTER + 27, shape.top - 18),
                    (CENTER, shape.top + 4),
                ],
                samples=12,
            ),
            dark,
        )


def _draw_hair_front(
    fill: Painter, shape: FaceShape, attributes: AvatarAttributes, hair: Rgb
) -> None:
    style = attributes.hair_style
    light = shade(hair, 0.16)
    if style == "bald":
        return
    if style == "buzz":
        fill(_hair_cap(shape, volume=2, side_y=126, hairline=100), hair)
        return
    if style == "undercut":
        fill(_hair_cap(shape, volume=15, side_y=104, hairline=96), hair)
        return
    if style == "curly":
        fill(_hair_cap(shape, volume=14, side_y=142, hairline=100), hair)
        for index, (cx, cy) in enumerate(
            ellipse_points((CENTER, shape.top + 34), shape.temple + 6, 42, 186, 354, steps=8)
        ):
            radius = 15.0 if index % 2 else 12.0
            fill(
                catmull_rom(
                    [(cx - radius, cy), (cx, cy - radius), (cx + radius, cy), (cx, cy + radius)],
                    samples=10,
                ),
                hair if index % 2 else light,
            )
        return
    if style == "afro":
        fill(_hair_cap(shape, volume=10, side_y=140, hairline=104), hair)
        return
    if style == "wavy":
        fill(_hair_cap(shape, volume=18, side_y=150, hairline=100), hair)
        fill(
            catmull_rom(
                [
                    (CENTER - shape.temple + 4, 98.0),
                    (CENTER - 24, 110.0),
                    (CENTER + 8, 94.0),
                    (CENTER + shape.temple - 6, 108.0),
                    (CENTER + shape.temple - 2, 82.0),
                    (CENTER - shape.temple + 6, 78.0),
                ],
                samples=10,
            ),
            shade(hair, 0.09),
        )
        return
    if style == "side-parted":
        fill(_hair_cap(shape, volume=14, side_y=142, hairline=99), hair)
        fill(
            catmull_rom(
                [
                    (CENTER - 30, 92.0),
                    (CENTER + shape.temple - 4, 82.0),
                    (CENTER + shape.temple + 2, 112.0),
                    (CENTER + 14, 96.0),
                ],
                samples=10,
            ),
            light,
        )
        return
    if style in {"long", "bob", "ponytail", "bun"}:
        side = 130.0 if style == "bun" else 150.0
        fill(_hair_cap(shape, volume=13, side_y=side, hairline=98), hair)
        return
    # short
    fill(_hair_cap(shape, volume=12, side_y=138, hairline=100), hair)
    fill(
        catmull_rom(
            [
                (CENTER - 34, 92.0),
                (CENTER + 6, 82.0),
                (CENTER + shape.temple - 8, 96.0),
                (CENTER - 6, 96.0),
            ],
            samples=10,
        ),
        light,
    )


def _facial_hair_color(attributes: AvatarAttributes, hair: Rgb, skin: Rgb) -> Rgb:
    return mix(skin, hair, 0.45) if attributes.facial_hair == "stubble" else shade(hair, -0.04)


def _draw_beard(
    fill: Painter, shape: FaceShape, attributes: AvatarAttributes, hair: Rgb, skin: Rgb
) -> None:
    style = attributes.facial_hair
    if style == "none":
        return
    color = _facial_hair_color(attributes, hair, skin)
    if style in {"stubble", "short beard", "full beard"}:
        top = {"stubble": 150.0, "short beard": 146.0, "full beard": 138.0}[style]
        # Un solo recorrido: mejilla izquierda, mandíbula, mentón, mandíbula
        # derecha y vuelta por el borde interior. Saltar de un lado a otro
        # cerraría el polígono sobre sí mismo y dibujaría una banda recta.
        points: list[Point] = [
            (CENTER - shape.cheek + 4, top),
            (CENTER - shape.jaw - 1, 178.0),
            (CENTER - shape.chin - 8, shape.chin_y - 10),
            (CENTER, shape.chin_y - 1),
            (CENTER + shape.chin + 8, shape.chin_y - 10),
            (CENTER + shape.jaw + 1, 178.0),
            (CENTER + shape.cheek - 4, top),
            (CENTER + shape.cheek - 17, top + 18),
            (CENTER + 31, MOUTH_Y - (16.0 if style == "full beard" else 2.0)),
            (CENTER, MOUTH_Y + 13),
            (CENTER - 31, MOUTH_Y - (16.0 if style == "full beard" else 2.0)),
            (CENTER - shape.cheek + 17, top + 18),
        ]
        fill(catmull_rom(points, samples=8), color)
    elif style == "goatee":
        fill(
            catmull_rom(
                [
                    (CENTER - 21, MOUTH_Y + 4),
                    (CENTER, MOUTH_Y - 2),
                    (CENTER + 21, MOUTH_Y + 4),
                    (CENTER + 15, shape.chin_y - 8),
                    (CENTER, shape.chin_y - 2),
                    (CENTER - 15, shape.chin_y - 8),
                ],
                samples=10,
            ),
            color,
        )

def _draw_mustache(
    fill: Painter, attributes: AvatarAttributes, hair: Rgb, skin: Rgb
) -> None:
    """Capa posterior a la boca: el bigote se apoya sobre el labio superior."""
    if attributes.facial_hair not in {"mustache", "goatee", "full beard"}:
        return
    fill(
        catmull_rom(
            [
                (CENTER - 28, MOUTH_Y - 17),
                (CENTER, MOUTH_Y - 22),
                (CENTER + 28, MOUTH_Y - 17),
                (CENTER + 20, MOUTH_Y - 4),
                (CENTER, MOUTH_Y - 10),
                (CENTER - 20, MOUTH_Y - 4),
            ],
            samples=10,
        ),
        _facial_hair_color(attributes, hair, skin),
    )


def _eye_geometry(shape: str) -> tuple[float, float, float, float]:
    """Semiejes de la abertura, radio del iris y descenso del párpado."""
    return {
        "almond": (18.0, 10.5, 8.5, 0.0),
        "round": (15.5, 14.0, 9.5, 0.0),
        "narrow": (18.5, 7.5, 8.0, 0.0),
        "wide": (19.5, 13.0, 9.5, 0.0),
        "hooded": (18.0, 11.0, 8.5, 4.0),
    }[shape]


def _brow_points(style: str, sign: float) -> list[Point]:
    inner = CENTER + sign * 12
    outer = CENTER + sign * 44
    thickness, tilt, peak = {
        "natural": (7.0, 2.0, -2.0),
        "arched": (6.0, 1.0, -7.0),
        "thick": (10.5, 2.0, -2.0),
        "thin": (4.0, 2.0, -2.0),
        "angled": (7.0, 7.0, -1.0),
    }[style]
    top: list[Point] = [
        (inner, BROW_Y + tilt),
        (CENTER + sign * 28, BROW_Y + peak),
        (outer, BROW_Y + 1),
    ]
    bottom: list[Point] = [
        (outer, BROW_Y + 1 + thickness * 0.55),
        (CENTER + sign * 28, BROW_Y + peak + thickness),
        (inner, BROW_Y + tilt + thickness * 0.8),
    ]
    return catmull_rom([*top, *bottom], samples=8)


@dataclass(frozen=True, slots=True)
class FlatVectorAvatarRenderer:
    """Dibuja el avatar desde los atributos, sin modelo neuronal (ADR 0012).

    Las coordenadas están sobre un lienzo de referencia de 256 px y se escalan
    al tamaño pedido; el rostro, el pelo y la barba se describen con splines
    para que las siluetas sean orgánicas y no un montaje de rectángulos.
    """

    image_size: int = 256

    def render(self, attributes: AvatarAttributes) -> Image.Image:
        factor = self.image_size * SUPERSAMPLE / 256
        canvas = round(256 * factor)
        background = hex_to_rgb(BACKGROUNDS[attributes.background])
        image = Image.new("RGB", (canvas, canvas), background)
        draw = ImageDraw.Draw(image)

        def fill(points: Sequence[Point], color: Rgb) -> None:
            draw.polygon([(x * factor, y * factor) for x, y in points], fill=color)

        def stroke(points: Sequence[Point], color: Rgb, width: float) -> None:
            draw.line(
                [(x * factor, y * factor) for x, y in points],
                fill=color,
                width=max(1, round(width * factor)),
                joint="curve",
            )

        def box(x0: float, y0: float, x1: float, y1: float) -> list[float]:
            return [x0 * factor, y0 * factor, x1 * factor, y1 * factor]

        def frame_width(value: float) -> int:
            return max(1, round(value * factor))

        shape = FACE_SHAPES[attributes.face_shape]
        skin = hex_to_rgb(SKIN_TONES[attributes.skin_tone])
        skin_shadow = shade(skin, -0.09)
        hair = hex_to_rgb(HAIR_COLORS[attributes.hair_color])
        cloth = hex_to_rgb(CLOTHING_COLORS[attributes.clothing_color])

        _draw_hair_back(fill, shape, attributes, hair)
        self._draw_body(fill, shape, attributes, cloth, skin, skin_shadow)
        for sign in (-1.0, 1.0):
            cx = CENTER + sign * (shape.cheek - 1)
            fill(
                catmull_rom(
                    [(cx - 9, 144.0), (cx, 132.0), (cx + 9, 148.0), (cx, 168.0)], samples=10
                ),
                skin,
            )
        fill(face_outline(shape), skin)
        fill(
            catmull_rom(
                [
                    (CENTER - shape.temple + 2, 96.0),
                    (CENTER - shape.cheek + 1, 140.0),
                    (CENTER - shape.jaw + 2, 180.0),
                    (CENTER - shape.chin, shape.chin_y - 10),
                    (CENTER - shape.jaw + 17, 176.0),
                    (CENTER - shape.cheek + 19, 136.0),
                ],
                samples=10,
            ),
            skin_shadow,
        )

        self._draw_eyes(fill, draw, box, attributes)
        self._draw_nose(fill, stroke, attributes, skin_shadow)
        _draw_beard(fill, shape, attributes, hair, skin)
        self._draw_mouth(fill, attributes)
        _draw_mustache(fill, attributes, hair, skin)
        self._draw_freckles(draw, box, attributes, skin_shadow)
        _draw_hair_front(fill, shape, attributes, hair)
        self._draw_glasses(draw, box, stroke, frame_width, shape, attributes)
        self._draw_earrings(draw, box, frame_width, shape, attributes)

        return image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

    @staticmethod
    def _draw_body(
        fill: Painter,
        shape: FaceShape,
        attributes: AvatarAttributes,
        cloth: Rgb,
        skin: Rgb,
        skin_shadow: Rgb,
    ) -> None:
        fill(
            [
                (CENTER - 23, shape.chin_y - 26),
                (CENTER + 23, shape.chin_y - 26),
                (CENTER + 25, shape.chin_y + 26),
                (CENTER - 25, shape.chin_y + 26),
            ],
            skin_shadow,
        )
        garment = attributes.clothing
        top_y = {
            "crew neck": 220.0,
            "v-neck": 216.0,
            "collared shirt": 218.0,
            "hoodie": 212.0,
            "turtleneck": 204.0,
        }[garment]
        shoulders: list[Point] = [
            (CENTER - 118, 258.0),
            (CENTER - 94, 234.0),
            (CENTER - 44, 220.0),
            (CENTER, 216.0),
            (CENTER + 44, 220.0),
            (CENTER + 94, 234.0),
            (CENTER + 118, 258.0),
        ]
        body = [
            (x, max(y, top_y)) for x, y in catmull_rom(shoulders, samples=10, closed=False)
        ]
        fill([*body, (CENTER + 132, 264.0), (CENTER - 132, 264.0)], cloth)
        if garment == "v-neck":
            fill([(CENTER - 22, top_y - 1), (CENTER, top_y + 30), (CENTER + 22, top_y - 1)], skin)
        elif garment == "collared shirt":
            highlight = shade(cloth, 0.22)
            fill([(CENTER - 27, top_y - 2), (CENTER, top_y + 26), (CENTER - 5, top_y - 2)],
                 highlight)
            fill([(CENTER + 27, top_y - 2), (CENTER, top_y + 26), (CENTER + 5, top_y - 2)],
                 highlight)
        elif garment == "hoodie":
            fill(
                catmull_rom(
                    [
                        (CENTER - 62, top_y + 6),
                        (CENTER, top_y + 32),
                        (CENTER + 62, top_y + 6),
                        (CENTER, top_y - 8),
                    ],
                    samples=10,
                ),
                shade(cloth, -0.16),
            )
        elif garment == "turtleneck":
            fill(
                [
                    (CENTER - 29, top_y - 10),
                    (CENTER + 29, top_y - 10),
                    (CENTER + 31, top_y + 16),
                    (CENTER - 31, top_y + 16),
                ],
                shade(cloth, 0.13),
            )

    @staticmethod
    def _draw_eyes(
        fill: Painter, draw: ImageDraw.ImageDraw, box: BoxFn, attributes: AvatarAttributes
    ) -> None:
        half_w, half_h, iris_r, hood = _eye_geometry(attributes.eye_shape)
        iris = hex_to_rgb(EYE_COLORS[attributes.eye_color])
        dark = hex_to_rgb(LINE)
        white = hex_to_rgb(SCLERA)
        for sign in (-1.0, 1.0):
            cx = CENTER + sign * EYE_DX
            draw.ellipse(box(cx - half_w, EYE_Y - half_h, cx + half_w, EYE_Y + half_h), fill=white)
            draw.ellipse(box(cx - iris_r, EYE_Y - iris_r, cx + iris_r, EYE_Y + iris_r), fill=iris)
            draw.ellipse(box(cx - 4.2, EYE_Y - 4.2, cx + 4.2, EYE_Y + 4.2), fill=dark)
            draw.ellipse(box(cx + 1.5, EYE_Y - 6.5, cx + 6.5, EYE_Y - 1.5), fill=white)
            # Párpado superior: cierra la forma y da carácter a la mirada.
            fill(
                catmull_rom(
                    [
                        (cx - half_w - 1, EYE_Y - half_h * 0.35),
                        (cx, EYE_Y - half_h - 1.5 + hood),
                        (cx + half_w + 1, EYE_Y - half_h * 0.35),
                        (cx, EYE_Y - half_h + 4.0 + hood),
                    ],
                    samples=8,
                ),
                dark,
            )
            fill(_brow_points(attributes.brow_style, sign), dark)

    @staticmethod
    def _draw_nose(
        fill: Painter,
        stroke: LinePainter,
        attributes: AvatarAttributes,
        skin_shadow: Rgb,
    ) -> None:
        style = attributes.nose_style
        width, tip = {
            "straight": (9.0, NOSE_BOTTOM),
            "small": (7.5, NOSE_BOTTOM - 6),
            "button": (9.5, NOSE_BOTTOM - 4),
            "wide": (13.5, NOSE_BOTTOM),
            "pointed": (7.0, NOSE_BOTTOM + 2),
        }[style]
        if style == "button":
            fill(
                catmull_rom(
                    [
                        (CENTER - width, tip - 3),
                        (CENTER, tip - 11),
                        (CENTER + width, tip - 3),
                        (CENTER, tip + 4),
                    ],
                    samples=10,
                ),
                skin_shadow,
            )
            return
        stroke(
            catmull_rom(
                [
                    (CENTER + 2.5, NOSE_TOP),
                    (CENTER + 3.0, tip - 8),
                    (CENTER - width * 0.5, tip),
                    (CENTER - width, tip - 2),
                ],
                samples=8,
                closed=False,
            ),
            skin_shadow,
            3.4,
        )

    @staticmethod
    def _draw_mouth(fill: Painter, attributes: AvatarAttributes) -> None:
        lip = hex_to_rgb(LIP)
        expression = attributes.expression
        if expression in {"happy", "smiling"}:
            open_mouth = expression == "happy"
            fill(
                catmull_rom(
                    [
                        (CENTER - 25, MOUTH_Y - 4),
                        (CENTER, MOUTH_Y + (16 if open_mouth else 10)),
                        (CENTER + 25, MOUTH_Y - 4),
                        (CENTER, MOUTH_Y - 7),
                    ],
                    samples=10,
                ),
                lip,
            )
            if open_mouth:
                fill(
                    catmull_rom(
                        [
                            (CENTER - 17, MOUTH_Y - 3),
                            (CENTER, MOUTH_Y + 3),
                            (CENTER + 17, MOUTH_Y - 3),
                            (CENTER, MOUTH_Y - 5),
                        ],
                        samples=10,
                    ),
                    hex_to_rgb(TEETH),
                )
        elif expression == "friendly":
            fill(
                catmull_rom(
                    [
                        (CENTER - 22, MOUTH_Y - 3),
                        (CENTER, MOUTH_Y + 8),
                        (CENTER + 22, MOUTH_Y - 3),
                        (CENTER, MOUTH_Y + 1),
                    ],
                    samples=10,
                ),
                lip,
            )
        elif expression == "confident":
            fill(
                catmull_rom(
                    [
                        (CENTER - 22, MOUTH_Y + 2),
                        (CENTER + 4, MOUTH_Y + 6),
                        (CENTER + 22, MOUTH_Y - 4),
                        (CENTER + 2, MOUTH_Y + 1),
                    ],
                    samples=10,
                ),
                lip,
            )
        elif expression == "serious":
            fill(
                [
                    (CENTER - 20, MOUTH_Y - 1),
                    (CENTER + 20, MOUTH_Y - 1),
                    (CENTER + 20, MOUTH_Y + 3),
                    (CENTER - 20, MOUTH_Y + 3),
                ],
                lip,
            )
        else:  # calm
            fill(
                catmull_rom(
                    [
                        (CENTER - 20, MOUTH_Y),
                        (CENTER - 7, MOUTH_Y - 4),
                        (CENTER, MOUTH_Y - 1),
                        (CENTER + 7, MOUTH_Y - 4),
                        (CENTER + 20, MOUTH_Y),
                        (CENTER + 8, MOUTH_Y + 6),
                        (CENTER, MOUTH_Y + 7),
                        (CENTER - 8, MOUTH_Y + 6),
                    ],
                    samples=10,
                ),
                lip,
            )

    @staticmethod
    def _draw_freckles(
        draw: ImageDraw.ImageDraw,
        box: BoxFn,
        attributes: AvatarAttributes,
        skin_shadow: Rgb,
    ) -> None:
        density = attributes.effective_freckles
        if density == "none":
            return
        color = shade(skin_shadow, -0.22)
        rows = (0, 8) if density == "heavy" else (0,)
        for row in rows:
            for index in range(4):
                for sign in (-1.0, 1.0):
                    x = CENTER + sign * (22 + index * 9)
                    y = NOSE_TOP + 6 + row + (index % 2) * 5
                    draw.ellipse(box(x - 2.2, y - 2.2, x + 2.2, y + 2.2), fill=color)

    @staticmethod
    def _draw_glasses(
        draw: ImageDraw.ImageDraw,
        box: BoxFn,
        stroke: LinePainter,
        frame_width: Callable[[float], int],
        shape: FaceShape,
        attributes: AvatarAttributes,
    ) -> None:
        style = attributes.effective_glasses
        if style == "none":
            return
        frame = hex_to_rgb(LINE)
        for sign in (-1.0, 1.0):
            cx = CENTER + sign * EYE_DX
            if style == "round":
                draw.ellipse(
                    box(cx - 24, EYE_Y - 22, cx + 24, EYE_Y + 22),
                    outline=frame,
                    width=frame_width(3.5),
                )
            elif style == "rectangular":
                draw.rounded_rectangle(
                    box(cx - 26, EYE_Y - 15, cx + 26, EYE_Y + 15),
                    radius=frame_width(5),
                    outline=frame,
                    width=frame_width(3.5),
                )
            elif style == "square":
                draw.rounded_rectangle(
                    box(cx - 25, EYE_Y - 20, cx + 25, EYE_Y + 20),
                    radius=frame_width(8),
                    outline=frame,
                    width=frame_width(3.5),
                )
            else:  # sunglasses
                draw.rounded_rectangle(
                    box(cx - 26, EYE_Y - 19, cx + 26, EYE_Y + 17), radius=frame_width(10),
                    fill=frame,
                )
        stroke([(CENTER - 6, EYE_Y - 2), (CENTER + 6, EYE_Y - 2)], frame, 3.5)
        stroke([(CENTER - shape.cheek, EYE_Y - 7), (CENTER - 50, EYE_Y - 3)], frame, 3.5)
        stroke([(CENTER + 50, EYE_Y - 3), (CENTER + shape.cheek, EYE_Y - 7)], frame, 3.5)

    @staticmethod
    def _draw_earrings(
        draw: ImageDraw.ImageDraw,
        box: BoxFn,
        frame_width: Callable[[float], int],
        shape: FaceShape,
        attributes: AvatarAttributes,
    ) -> None:
        style = attributes.effective_earrings
        if style == "none":
            return
        gold = hex_to_rgb(GOLD)
        for sign in (-1.0, 1.0):
            cx = CENTER + sign * (shape.cheek + 5)
            if style == "studs":
                draw.ellipse(box(cx - 5, 164, cx + 5, 174), fill=gold)
            else:
                draw.ellipse(
                    box(cx - 10, 164, cx + 10, 190), outline=gold, width=frame_width(3.5)
                )
