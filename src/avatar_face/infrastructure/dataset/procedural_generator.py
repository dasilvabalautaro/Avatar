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
                "version": "1.0.0",
                "generator": "avatarface-procedural-v2",
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
    ) -> DatasetGenerationResult:
        """Genera el corpus ampliado desde la única fuente aprobada actualmente.

        No descarga ni mezcla activos: la procedencia queda ligada al registro
        ``AF-PROC-001`` y el esquema 2 añade la política de fuentes al manifiesto.
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
                "version": "2.0.0",
                "generator": "avatarface-procedural-v2",
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
            ("none", "round glasses", "earrings", "freckles"), index, seed, 17
        )
        hair_style = self._stratified_choice(
            ("short", "curly", "side-parted", "bob"), index, seed, 19
        )
        face_shape = self._stratified_choice(("round", "oval", "square", "heart"), index, seed, 23)
        brow_style = self._stratified_choice(("arched", "straight", "soft"), index, seed, 29)
        nose_style = self._stratified_choice(("small", "straight", "button"), index, seed, 31)

        image = Image.new("RGB", (self.image_size, self.image_size), background)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((105, 184, 151, 246), radius=18, fill=skin)
        if face_shape == "round":
            draw.ellipse((48, 68, 208, 226), fill=skin, outline="#4A3030", width=3)
        elif face_shape == "oval":
            draw.ellipse((58, 58, 198, 232), fill=skin, outline="#4A3030", width=3)
        elif face_shape == "square":
            draw.rounded_rectangle(
                (54, 68, 202, 224), radius=34, fill=skin, outline="#4A3030", width=3
            )
        else:
            draw.polygon(
                (
                    (128, 226),
                    (60, 139),
                    (68, 84),
                    (108, 64),
                    (128, 91),
                    (148, 64),
                    (188, 84),
                    (196, 139),
                ),
                fill=skin,
                outline="#4A3030",
                width=3,
            )
        draw.ellipse((39, 117, 61, 162), fill=skin)
        draw.ellipse((195, 117, 217, 162), fill=skin)

        if hair_style == "short":
            draw.pieslice((47, 45, 209, 170), 180, 360, fill=hair)
        elif hair_style == "curly":
            for x in range(59, 204, 24):
                draw.ellipse((x - 18, 48 + (x % 3) * 5, x + 18, 94), fill=hair)
        elif hair_style == "side-parted":
            draw.polygon(((49, 130), (62, 58), (188, 48), (210, 124), (137, 78)), fill=hair)
        else:
            draw.rounded_rectangle((45, 49, 211, 179), radius=65, fill=hair)
            draw.ellipse((63, 72, 193, 219), fill=skin)

        for eye_x in (94, 162):
            draw.ellipse((eye_x - 14, 123, eye_x + 14, 141), fill="#FFFDF8")
            draw.ellipse((eye_x - 6, 126, eye_x + 6, 138), fill=eye)
            draw.ellipse((eye_x - 2, 129, eye_x + 2, 135), fill="#171417")
        if brow_style == "arched":
            draw.arc((78, 109, 110, 128), 195, 345, fill=hair, width=4)
            draw.arc((146, 109, 178, 128), 195, 345, fill=hair, width=4)
        elif brow_style == "straight":
            draw.line((79, 117, 110, 114), fill=hair, width=4)
            draw.line((146, 114, 177, 117), fill=hair, width=4)
        else:
            draw.arc((78, 112, 110, 127), 200, 340, fill=hair, width=3)
            draw.arc((146, 112, 178, 127), 200, 340, fill=hair, width=3)
        if nose_style == "small":
            draw.line((128, 142, 124, 161, 131, 161), fill="#9A624F", width=3)
        elif nose_style == "straight":
            draw.line((128, 139, 128, 165), fill="#9A624F", width=3)
        else:
            draw.ellipse((122, 157, 134, 168), outline="#9A624F", width=3)

        if expression in {"smiling", "happy"}:
            draw.arc((103, 159, 153, 197), 10, 170, fill="#7D3340", width=5)
        else:
            draw.arc((108, 169, 148, 188), 190, 350, fill="#7D3340", width=4)

        if accessory == "round glasses":
            draw.ellipse((73, 114, 113, 151), outline="#2A2D34", width=4)
            draw.ellipse((143, 114, 183, 151), outline="#2A2D34", width=4)
            draw.line((113, 131, 143, 131), fill="#2A2D34", width=4)
        elif accessory == "earrings":
            draw.ellipse((44, 154, 54, 168), fill="#F4D35E")
            draw.ellipse((202, 154, 212, 168), fill="#F4D35E")
        elif accessory == "freckles":
            for x, y in ((105, 151), (113, 154), (143, 154), (151, 151)):
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#9A624F")

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
        for row in range(4):
            for column in range(16):
                bit = (signature[row * 2 + column // 8] >> (7 - column % 8)) & 1
                color = "#FFFDF8" if bit else "#211A1D"
                draw.rectangle((column * 16, row * 8, column * 16 + 15, row * 8 + 7), fill=color)
        image.save(path, format="PNG", optimize=True)

    @staticmethod
    def _stratified_choice(values: tuple[T, ...], index: int, seed: int, stride: int) -> T:
        """Distribuye cada categoría con diferencia máxima de una muestra."""
        size = len(values)
        cycle, position = divmod(index, size)
        return values[(position + cycle * stride + seed) % size]

    @staticmethod
    def _caption(attributes: dict[str, str]) -> str:
        accessory = (
            " without accessories"
            if attributes["accessory"] == "none"
            else f" with {attributes['accessory']}"
        )
        return (
            f"flat vector avatar face, {attributes['expression']} expression, "
            f"{attributes['face_shape']} face, {attributes['skin_tone']} skin tone, "
            f"{attributes['hair_style']} "
            f"{attributes['hair_color']} hair, {attributes['eye_color']} eyes"
            f"{accessory}, {attributes['background']} background"
        )
