from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from PIL import Image

from avatar_face.domain.dataset import DatasetAuditResult, DatasetSample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _perceptual_signature(path: Path) -> tuple[int, tuple[int, ...]]:
    """Firma RGB compacta: hash de estructura y miniatura para confirmar similitud."""
    with Image.open(path) as image:
        reduced = image.convert("RGB").resize((16, 16), Image.Resampling.LANCZOS)
        pixels = cast(list[tuple[int, int, int]], list(reduced.get_flattened_data()))
    channels = tuple(tuple(pixel[channel] for pixel in pixels) for channel in range(3))
    perceptual_hash = sum(
        1 << (channel * 64 + index)
        for channel, values in enumerate(channels)
        for index, value in enumerate(values[:64])
        if value >= sum(values) / len(values)
    )
    return perceptual_hash, tuple(value for pixel in pixels for value in pixel)


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _mean_pixel_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    return sum(abs(first - second) for first, second in zip(left, right, strict=True)) / len(left)


@dataclass(frozen=True, slots=True)
class JsonDatasetAuditor:
    """Audita manifiestos JSON y los archivos referenciados."""

    def audit(self, manifest_path: Path) -> DatasetAuditResult:
        manifest = manifest_path.expanduser().resolve()
        findings: list[str] = []
        if not manifest.is_file():
            return DatasetAuditResult(str(manifest), False, 0, 0, ("Manifiesto inexistente.",))
        payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return DatasetAuditResult(str(manifest), False, 0, 0, ("schema_version debe ser 1.",))
        metadata = payload.get("dataset")
        if not isinstance(metadata, dict):
            findings.append("Faltan metadatos del dataset.")
        else:
            if metadata.get("contains_real_people") is not False:
                findings.append("El smoke dataset no puede contener personas reales.")
            if metadata.get("uses_external_assets") is not False:
                findings.append("El smoke dataset no puede usar activos externos.")
            if metadata.get("license_id") != "CC0-1.0":
                findings.append("El smoke dataset debe declarar CC0-1.0.")

        raw_samples = payload.get("samples")
        if not isinstance(raw_samples, list):
            return DatasetAuditResult(
                str(manifest), False, 0, 0, tuple(findings + ["samples debe ser una lista."])
            )

        hashes = []
        perceptual_hashes: list[tuple[str, int, tuple[int, ...]]] = []
        identifiers = set()
        for index, raw in enumerate(raw_samples):
            if not isinstance(raw, dict):
                findings.append(f"Muestra {index}: registro inválido.")
                continue
            try:
                attributes = raw.get("attributes", {})
                if not isinstance(attributes, dict):
                    raise ValueError("attributes debe ser un objeto.")
                sample = DatasetSample(
                    identifier=str(raw["identifier"]),
                    image=str(raw["image"]),
                    caption=str(raw["caption"]),
                    attributes=tuple(sorted((str(k), str(v)) for k, v in attributes.items())),
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
                findings.append(f"Muestra {index}: {error}")
                continue
            if sample.identifier in identifiers:
                findings.append(f"ID duplicado: {sample.identifier}.")
            identifiers.add(sample.identifier)
            image_path = manifest.parent / sample.image
            if not image_path.is_file():
                findings.append(f"Archivo inexistente: {sample.image}.")
                continue
            actual_hash = _sha256(image_path)
            if actual_hash != sample.sha256:
                findings.append(f"Hash no coincide: {sample.image}.")
            hashes.append(actual_hash)
            perceptual_hash, thumbnail = _perceptual_signature(image_path)
            perceptual_hashes.append((sample.image, perceptual_hash, thumbnail))

        unique_hashes = len(set(hashes))
        if unique_hashes != len(hashes):
            findings.append("Existen imágenes duplicadas por SHA-256.")
        for left, right, distance in _similar_pairs(perceptual_hashes):
            findings.append(
                f"Similitud perceptual excesiva ({distance} bits): {left} y {right}."
            )
        return DatasetAuditResult(
            str(manifest), not findings, len(raw_samples), unique_hashes, tuple(findings)
        )


def _similar_pairs(
    hashes: list[tuple[str, int, tuple[int, ...]]],
) -> tuple[tuple[str, str, int], ...]:
    """Compara sólo candidatos que comparten una banda del pHash RGB."""
    buckets: dict[tuple[int, int], list[tuple[str, int, tuple[int, ...]]]] = {}
    pairs: set[tuple[str, str, int]] = set()
    for path, value, thumbnail in hashes:
        candidates: dict[str, tuple[int, tuple[int, ...]]] = {}
        for band in range(12):
            key = (band, (value >> (band * 16)) & 0xFFFF)
            candidates.update(
                {
                    candidate_path: (candidate_value, candidate_thumbnail)
                    for candidate_path, candidate_value, candidate_thumbnail in buckets.get(key, [])
                }
            )
            buckets.setdefault(key, []).append((path, value, thumbnail))
        for candidate_path, (candidate_value, candidate_thumbnail) in candidates.items():
            distance = _hamming_distance(value, candidate_value)
            if distance <= 3 and _mean_pixel_distance(thumbnail, candidate_thumbnail) <= 2.0:
                pairs.add((candidate_path, path, distance))
    return tuple(sorted(pairs))
