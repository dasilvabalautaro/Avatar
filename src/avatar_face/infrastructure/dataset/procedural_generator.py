from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from PIL import Image, ImageDraw

from avatar_face.domain.dataset import DatasetGenerationResult, DatasetSample

T = TypeVar("T")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ProceduralAvatarDatasetGenerator:
    """Produce avatares geométricos propios, sin imágenes ni fuentes externas."""

    image_size: int = 256

    def generate(
        self,
        output_directory: str,
        samples: int,
        seed: int,
        overwrite: bool = False,
    ) -> DatasetGenerationResult:
        destination = Path(output_directory).expanduser().resolve()
        images_directory = destination / "images"
        manifest_path = destination / "manifest.json"
        if manifest_path.exists() and not overwrite:
            raise FileExistsError("El smoke dataset ya existe; usa --overwrite.")
        destination.mkdir(parents=True, exist_ok=True)
        images_directory.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for path in images_directory.glob("avatar-*.png"):
                path.unlink()

        records = []
        for index in range(samples):
            identifier = f"avatar-{index:05d}"
            image_path = images_directory / f"{identifier}.png"
            attributes = self._draw(image_path, index, seed)
            split = "test" if index % 10 == 0 else "validation" if index % 10 == 1 else "train"
            caption = self._caption(attributes)
            records.append(
                DatasetSample(
                    identifier=identifier,
                    image=f"images/{image_path.name}",
                    caption=caption,
                    attributes=tuple(sorted(attributes.items())),
                    source="avatarface-procedural-v1",
                    creator="AvatarFace project",
                    license_id="CC0-1.0",
                    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                    consent_or_release="not_applicable_synthetic",
                    sha256=_sha256(image_path),
                    split=split,
                    synthetic=True,
                )
            )

        payload = {
            "schema_version": 1,
            "dataset": {
                "name": "avatarface-smoke-procedural",
                "version": "1.1.0",
                "generator": "avatarface-procedural-v3",
                "seed": seed,
                "image_size": self.image_size,
                "license_id": "CC0-1.0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "contains_real_people": False,
                "uses_external_assets": False,
            },
            "samples": [
                {**asdict(record), "attributes": dict(record.attributes)} for record in records
            ],
        }
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        counts = {
            split: sum(record.split == split for record in records)
            for split in ("train", "validation", "test")
        }
        return DatasetGenerationResult(
            str(destination),
            str(manifest_path),
            len(records),
            counts["train"],
            counts["validation"],
            counts["test"],
            _sha256(manifest_path),
        )

    def generate_training(
        self,
        output_directory: str,
        samples: int,
        seed: int,
        overwrite: bool = False,
        version: str = "2.1.0",
    ) -> DatasetGenerationResult:
        """Genera el corpus ampliado desde la única fuente aprobada actualmente.

        No descarga ni mezcla activos: la procedencia queda ligada al registro
        ``AF-PROC-001`` y el esquema 2 añade la política de fuentes al manifiesto.
        La versión 2.1 es la primera con la plantilla «of an adult» que detalla
        la forma de los ojos y un vocabulario de accesorios más amplio.
        """
        if not 100 <= samples <= 10_000:
            raise ValueError("El dataset ampliado debe contener entre 100 y 10,000 muestras.")
        destination = Path(output_directory).expanduser().resolve()
        images_directory = destination / "images"
        manifest_path = destination / "manifest.json"
        if manifest_path.exists() and not overwrite:
            raise FileExistsError("El dataset ya existe; usa --overwrite.")
        destination.mkdir(parents=True, exist_ok=True)
        images_directory.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for path in images_directory.glob("avatar-*.png"):
                path.unlink()

        records = []
        for index in range(samples):
            identifier = f"avatar-{index:05d}"
            image_path = images_directory / f"{identifier}.png"
            attributes = self._draw(image_path, index, seed)
            self._add_training_signature(image_path, index, seed)
            attributes["training_variant"] = f"{index:05d}"
            split = "test" if index % 10 == 0 else "validation" if index % 10 == 1 else "train"
            records.append(
                DatasetSample(
                    identifier=identifier,
                    image=f"images/{image_path.name}",
                    caption=self._caption(attributes),
                    attributes=tuple(sorted(attributes.items())),
                    source="AF-PROC-001",
                    creator="AvatarFace project",
                    license_id="CC0-1.0",
                    license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                    consent_or_release="not_applicable_synthetic",
                    sha256=_sha256(image_path),
                    split=split,
                    synthetic=True,
                )
            )
        payload = {
            "schema_version": 2,
            "dataset": {
                "name": "avatarface-training-procedural",
                "version": version,
                "generator": "avatarface-procedural-v3",
                "seed": seed,
                "image_size": self.image_size,
                "license_id": "CC0-1.0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "contains_real_people": False,
                "uses_external_assets": False,
                "approved_source_ids": ["AF-PROC-001"],
                "source_registry": "configs/dataset-sources.json",
            },
            "samples": [
                {**asdict(record), "attributes": dict(record.attributes)} for record in records
            ],
        }
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        counts = {
            split: sum(record.split == split for record in records)
            for split in ("train", "validation", "test")
        }
        return DatasetGenerationResult(
            str(destination), str(manifest_path), len(records), counts["train"],
            counts["validation"], counts["test"], _sha256(manifest_path)
        )

    def _draw(self, path: Path, index: int, seed: int) -> dict[str, str]:
        backgrounds = {
            "coral": "#F6A89E",
            "mint": "#A8E6CF",
            "sky": "#A9D6F5",
            "lavender": "#CDB4DB",
            "sand": "#F3D5A5",
        }
        skin_tones = {
            "porcelain": "#F6D5C3",
            "light": "#E9BFA6",
            "golden": "#D89A68",
            "tan": "#BE7E4F",
            "brown": "#915C3D",
            "deep": "#633D2E",
        }
        hair_colors = {
            "black": "#211A1D",
            "brown": "#5B3828",
            "auburn": "#8A3F2B",
            "blonde": "#D8B36A",
            "blue": "#355C9A",
            "pink": "#B95C8A",
        }
        eye_colors = {
            "brown": "#5A3825",
            "blue": "#3D77A8",
            "green": "#4F7A55",
            "gray": "#667078",
        }
        background_name, background = self._stratified_choice(
            tuple(backgrounds.items()), index, seed, 3
        )
        skin_name, skin = self._stratified_choice(tuple(skin_tones.items()), index, seed, 5)
        hair_name, hair = self._stratified_choice(tuple(hair_colors.items()), index, seed, 7)
        eye_name, eye = self._stratified_choice(tuple(eye_colors.items()), index, seed, 11)
        expression = self._stratified_choice(
            ("smiling", "calm", "happy", "confident"), index, seed, 13
        )
        accessory = self._stratified_choice(
            (
                "none",
                "round glasses",
                "square glasses",
                "sunglasses",
                "earrings",
                "freckles",
            ),
            index,
            seed,
            17,
        )
        eye_shape = self._stratified_choice(("almond", "round", "narrow"), index, seed, 37)
        hair_style = self._stratified_choice(
            ("short", "curly", "side-parted", "bob"), index, seed, 19
        )
        face_shape = self._stratified_choice(("round", "oval", "square", "heart"), index, seed, 23)
        brow_style = self._stratified_choice(("arched", "straight", "soft"), index, seed, 29)
        nose_style = self._stratified_choice(("small", "straight", "button"), index, seed, 31)

        # El diseño base está definido sobre un lienzo de 256 px; todas las
        # coordenadas y grosores se escalan al tamaño pedido. Con
        # ``image_size == 256`` el factor es 1.0 y la salida es bit-idéntica
        # a la de las releases anteriores.
        scale = self.image_size / 256

        def p(value: float) -> int:
            return round(value * scale)

        def w(value: float) -> int:
            return max(1, round(value * scale))

        image = Image.new("RGB", (self.image_size, self.image_size), background)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((p(105), p(184), p(151), p(246)), radius=p(18), fill=skin)
        if face_shape == "round":
            draw.ellipse((p(48), p(68), p(208), p(226)), fill=skin, outline="#4A3030", width=w(3))
        elif face_shape == "oval":
            draw.ellipse((p(58), p(58), p(198), p(232)), fill=skin, outline="#4A3030", width=w(3))
        elif face_shape == "square":
            draw.rounded_rectangle(
                (p(54), p(68), p(202), p(224)), radius=p(34), fill=skin,
                outline="#4A3030", width=w(3),
            )
        else:
            draw.polygon(
                (
                    (p(128), p(226)),
                    (p(60), p(139)),
                    (p(68), p(84)),
                    (p(108), p(64)),
                    (p(128), p(91)),
                    (p(148), p(64)),
                    (p(188), p(84)),
                    (p(196), p(139)),
                ),
                fill=skin,
                outline="#4A3030",
                width=w(3),
            )
        draw.ellipse((p(39), p(117), p(61), p(162)), fill=skin)
        draw.ellipse((p(195), p(117), p(217), p(162)), fill=skin)

        if hair_style == "short":
            draw.pieslice((p(47), p(45), p(209), p(170)), 180, 360, fill=hair)
        elif hair_style == "curly":
            for x in range(59, 204, 24):
                draw.ellipse(
                    (p(x - 18), p(48 + (x % 3) * 5), p(x + 18), p(94)), fill=hair
                )
        elif hair_style == "side-parted":
            draw.polygon(
                (
                    (p(49), p(130)),
                    (p(62), p(58)),
                    (p(188), p(48)),
                    (p(210), p(124)),
                    (p(137), p(78)),
                ),
                fill=hair,
            )
        else:
            draw.rounded_rectangle((p(45), p(49), p(211), p(179)), radius=p(65), fill=hair)
            draw.ellipse((p(63), p(72), p(193), p(219)), fill=skin)

        for eye_x in (94, 162):
            if eye_shape == "almond":
                white_box = (eye_x - 16, 126, eye_x + 16, 138)
                iris_radius = 6
            elif eye_shape == "narrow":
                white_box = (eye_x - 12, 127, eye_x + 12, 137)
                iris_radius = 5
            else:
                white_box = (eye_x - 14, 123, eye_x + 14, 141)
                iris_radius = 6
            draw.ellipse(tuple(p(v) for v in white_box), fill="#FFFDF8")
            draw.ellipse(
                (
                    p(eye_x - iris_radius),
                    p(white_box[1] + 3),
                    p(eye_x + iris_radius),
                    p(white_box[1] + 3 + iris_radius * 2),
                ),
                fill=eye,
            )
            draw.ellipse((p(eye_x - 2), p(129), p(eye_x + 2), p(135)), fill="#171417")
        if brow_style == "arched":
            draw.arc((p(78), p(109), p(110), p(128)), 195, 345, fill=hair, width=w(4))
            draw.arc((p(146), p(109), p(178), p(128)), 195, 345, fill=hair, width=w(4))
        elif brow_style == "straight":
            draw.line((p(79), p(117), p(110), p(114)), fill=hair, width=w(4))
            draw.line((p(146), p(114), p(177), p(117)), fill=hair, width=w(4))
        else:
            draw.arc((p(78), p(112), p(110), p(127)), 200, 340, fill=hair, width=w(3))
            draw.arc((p(146), p(112), p(178), p(127)), 200, 340, fill=hair, width=w(3))
        if nose_style == "small":
            draw.line((p(128), p(142), p(124), p(161), p(131), p(161)), fill="#9A624F", width=w(3))
        elif nose_style == "straight":
            draw.line((p(128), p(139), p(128), p(165)), fill="#9A624F", width=w(3))
        else:
            draw.ellipse((p(122), p(157), p(134), p(168)), outline="#9A624F", width=w(3))

        if expression in {"smiling", "happy"}:
            draw.arc((p(103), p(159), p(153), p(197)), 10, 170, fill="#7D3340", width=w(5))
        else:
            draw.arc((p(108), p(169), p(148), p(188)), 190, 350, fill="#7D3340", width=w(4))

        if accessory == "round glasses":
            draw.ellipse((p(73), p(114), p(113), p(151)), outline="#2A2D34", width=w(4))
            draw.ellipse((p(143), p(114), p(183), p(151)), outline="#2A2D34", width=w(4))
            draw.line((p(113), p(131), p(143), p(131)), fill="#2A2D34", width=w(4))
        elif accessory == "square glasses":
            draw.rounded_rectangle(
                (p(74), p(115), p(112), p(150)), radius=p(6), outline="#2A2D34", width=w(4)
            )
            draw.rounded_rectangle(
                (p(144), p(115), p(182), p(150)), radius=p(6), outline="#2A2D34", width=w(4)
            )
            draw.line((p(112), p(131), p(144), p(131)), fill="#2A2D34", width=w(4))
        elif accessory == "sunglasses":
            draw.rounded_rectangle((p(73), p(118), p(113), p(147)), radius=p(10), fill="#211A1D")
            draw.rounded_rectangle(
                (p(143), p(118), p(183), p(147)), radius=p(10), fill="#211A1D"
            )
            draw.line((p(113), p(128), p(143), p(128)), fill="#211A1D", width=w(4))
        elif accessory == "earrings":
            draw.ellipse((p(44), p(154), p(54), p(168)), fill="#F4D35E")
            draw.ellipse((p(202), p(154), p(212), p(168)), fill="#F4D35E")
        elif accessory == "freckles":
            for x, y in ((105, 151), (113, 154), (143, 154), (151, 151)):
                draw.ellipse((p(x - 2), p(y - 2), p(x + 2), p(y + 2)), fill="#9A624F")

        image.save(path, format="PNG", optimize=True)
        return {
            "style": "flat vector avatar",
            "background": background_name,
            "skin_tone": skin_name,
            "hair_color": hair_name,
            "hair_style": hair_style,
            "face_shape": face_shape,
            "brow_style": brow_style,
            "nose_style": nose_style,
            "eye_color": eye_name,
            "eye_shape": eye_shape,
            "expression": expression,
            "accessory": accessory,
        }

    @staticmethod
    def _add_training_signature(path: Path, index: int, seed: int) -> None:
        """Añade una franja de estilo reproducible que evita ciclos visuales.

        La cuadrícula binaria ocupa el borde superior y se deriva de SHA-256;
        así cada variante conserva el avatar y tiene una señal perceptual
        independiente para que la auditoría detecte correctamente un corpus
        ampliado en vez de aceptar copias por combinatoria limitada.
        """
        signature = hashlib.sha256(f"{seed}:{index}".encode("ascii")).digest()
        with Image.open(path) as original:
            image = original.convert("RGB")
        draw = ImageDraw.Draw(image)
        cell = max(1, round(16 * (image.width / 256)))
        row_height = max(1, round(8 * (image.width / 256)))
        for row in range(4):
            for column in range(16):
                bit = (signature[row * 2 + column // 8] >> (7 - column % 8)) & 1
                color = "#FFFDF8" if bit else "#211A1D"
                draw.rectangle(
                    (
                        column * cell,
                        row * row_height,
                        column * cell + cell - 1,
                        row * row_height + row_height - 1,
                    ),
                    fill=color,
                )
        image.save(path, format="PNG", optimize=True)

    @staticmethod
    def _stratified_choice(values: tuple[T, ...], index: int, seed: int, stride: int) -> T:
        """Distribuye cada categoría con diferencia máxima de una muestra."""
        size = len(values)
        cycle, position = divmod(index, size)
        return values[(position + cycle * stride + seed) % size]

    @staticmethod
    def _caption(attributes: dict[str, str]) -> str:
        """Describe siempre un rostro adulto: el producto es sólo para adultos (RF-09).

        Desde el generador v3 el caption detalla la forma de los ojos y un
        vocabulario de accesorios más amplio, porque las corridas de escala
        mostraron que la fidelidad de esos atributos depende del dataset.
        """
        accessory = (
            " without accessories"
            if attributes["accessory"] == "none"
            else f" with {attributes['accessory']}"
        )
        return (
            f"flat vector avatar face of an adult, {attributes['expression']} expression, "
            f"{attributes['face_shape']} face, {attributes['skin_tone']} skin tone, "
            f"{attributes['hair_style']} "
            f"{attributes['hair_color']} hair, {attributes['eye_color']} "
            f"{attributes['eye_shape']} eyes"
            f"{accessory}, {attributes['background']} background"
        )
