from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from avatar_face.domain.dataset import DatasetAuditResult, DatasetSample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

        unique_hashes = len(set(hashes))
        if unique_hashes != len(hashes):
            findings.append("Existen imágenes duplicadas por SHA-256.")
        return DatasetAuditResult(
            str(manifest), not findings, len(raw_samples), unique_hashes, tuple(findings)
        )
