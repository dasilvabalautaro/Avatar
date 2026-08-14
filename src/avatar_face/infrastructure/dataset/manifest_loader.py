from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor

from avatar_face.domain.dataset import DatasetLoadConfig, DatasetSample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_from_record(raw: object, index: int) -> DatasetSample:
    if not isinstance(raw, dict):
        raise ValueError(f"Muestra {index}: registro inválido.")
    attributes = raw.get("attributes", {})
    if not isinstance(attributes, dict):
        raise ValueError(f"Muestra {index}: attributes debe ser un objeto.")
    try:
        return DatasetSample(
            identifier=str(raw["identifier"]),
            image=str(raw["image"]),
            caption=str(raw["caption"]),
            attributes=tuple(sorted((str(key), str(value)) for key, value in attributes.items())),
            source=str(raw["source"]),
            creator=str(raw["creator"]),
            license_id=str(raw["license_id"]),
            license_url=str(raw["license_url"]),
            consent_or_release=str(raw["consent_or_release"]),
            sha256=str(raw["sha256"]),
            split=str(raw["split"]),
            synthetic=raw.get("synthetic") is True,
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"Muestra {index}: {error}") from error


@dataclass(frozen=True, slots=True)
class TorchDatasetBatch:
    """Batch de imágenes normalizadas y su evidencia de procedencia."""

    images: Tensor
    identifiers: tuple[str, ...]
    captions: tuple[str, ...]


class ManifestTorchDataset:
    """Carga sólo imágenes declaradas por el manifiesto, en orden reproducible."""

    def __init__(self, config: DatasetLoadConfig) -> None:
        self.config = config
        self.manifest_path = config.manifest_path.expanduser().resolve()
        self._samples = self._load_and_preflight()
        selected = [sample for sample in self._samples if sample.split == config.split]
        ordered = sorted(selected, key=lambda sample: sample.identifier)
        random.Random(config.seed).shuffle(ordered)
        self.samples = tuple(ordered)

    def _load_and_preflight(self) -> tuple[DatasetSample, ...]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifiesto inexistente: {self.manifest_path}")
        payload: Any = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2}:
            raise ValueError("schema_version debe ser 1 o 2.")
        raw_samples = payload.get("samples")
        if not isinstance(raw_samples, list):
            raise ValueError("samples debe ser una lista.")
        samples = tuple(_sample_from_record(raw, index) for index, raw in enumerate(raw_samples))
        identifiers = [sample.identifier for sample in samples]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("El manifiesto contiene IDs duplicados.")
        hashes = [sample.sha256 for sample in samples]
        if len(set(hashes)) != len(hashes):
            raise ValueError("El manifiesto contiene hashes de imagen duplicados.")
        for sample in samples:
            path = self.manifest_path.parent / sample.image
            if not path.is_file():
                raise FileNotFoundError(f"Archivo inexistente: {sample.image}")
            if _sha256(path) != sample.sha256:
                raise ValueError(f"Hash no coincide: {sample.image}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def batch(self, batch_index: int = 0) -> TorchDatasetBatch:
        if batch_index < 0:
            raise ValueError("batch_index no puede ser negativo.")
        start = batch_index * self.config.batch_size
        selected = self.samples[start : start + self.config.batch_size]
        if not selected:
            raise IndexError("El batch solicitado está fuera del split.")
        tensors = tuple(self._load_image(sample) for sample in selected)
        return TorchDatasetBatch(
            images=torch.stack(tensors),
            identifiers=tuple(sample.identifier for sample in selected),
            captions=tuple(sample.caption for sample in selected),
        )

    def _load_image(self, sample: DatasetSample) -> Tensor:
        path = self.manifest_path.parent / sample.image
        if self.config.verify_hashes_on_read and _sha256(path) != sample.sha256:
            raise ValueError(f"Hash no coincide al abrir: {sample.image}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            values = torch.tensor(bytearray(rgb.tobytes()), dtype=torch.uint8)
        return (
            values.reshape(height, width, 3)
            .permute(2, 0, 1)
            .to(torch.float32)
            .div_(127.5)
            .sub_(1.0)
        )
