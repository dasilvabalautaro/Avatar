from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from avatar_face.infrastructure.dataset.json_auditor import JsonDatasetAuditor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetRelease:
    lock_path: str
    manifest_sha256: str
    samples: int
    split_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class DatasetReleaseVerification:
    approved: bool
    manifest_sha256: str
    findings: tuple[str, ...]


def freeze_dataset(manifest_path: Path, output_path: Path, version: str) -> DatasetRelease:
    """Fija una versión inmutable mediante hashes de manifiesto y cada activo."""
    manifest = manifest_path.expanduser().resolve()
    audit = JsonDatasetAuditor().audit(manifest)
    if not audit.approved:
        raise ValueError(
            "La auditoría debe aprobarse antes de congelar: " + "; ".join(audit.findings)
        )
    if not version or any(character.isspace() for character in version):
        raise ValueError("La versión de release no puede ser vacía ni contener espacios.")
    payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
    samples = payload["samples"]
    image_hashes = {
        str(sample["image"]): str(sample["sha256"])
        for sample in sorted(samples, key=lambda item: str(item["image"]))
    }
    split_counts = {
        split: sum(sample["split"] == split for sample in samples)
        for split in ("train", "validation", "test")
    }
    manifest_sha256 = _sha256(manifest)
    lock = {
        "schema_version": 1,
        "release_version": version,
        "manifest_path": manifest.name,
        "manifest_sha256": manifest_sha256,
        "samples": len(samples),
        "split_counts": split_counts,
        "image_sha256": image_hashes,
        "audit": {"approved": True, "unique_hashes": audit.unique_hashes},
    }
    destination = output_path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError("El lock ya existe; una release congelada no se sobreescribe.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return DatasetRelease(str(destination), manifest_sha256, len(samples), split_counts)


def verify_frozen_dataset(manifest_path: Path, lock_path: Path) -> DatasetReleaseVerification:
    """Comprueba que un corpus aún coincide exactamente con su release congelada."""
    manifest = manifest_path.expanduser().resolve()
    lock_file = lock_path.expanduser().resolve()
    findings: list[str] = []
    audit = JsonDatasetAuditor().audit(manifest)
    if not audit.approved:
        findings.extend(audit.findings)
    if not lock_file.is_file():
        return DatasetReleaseVerification(False, "", ("Lock de release inexistente.",))
    lock: Any = json.loads(lock_file.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        return DatasetReleaseVerification(False, "", ("Lock de release inválido.",))
    actual_manifest_hash = _sha256(manifest) if manifest.is_file() else ""
    if lock.get("manifest_sha256") != actual_manifest_hash:
        findings.append("El hash del manifiesto no coincide con el lock.")
    if manifest.is_file():
        payload: Any = json.loads(manifest.read_text(encoding="utf-8"))
        samples = payload.get("samples", []) if isinstance(payload, dict) else []
        actual_images = {
            str(sample.get("image")): str(sample.get("sha256"))
            for sample in samples
            if isinstance(sample, dict)
        }
        if lock.get("image_sha256") != dict(sorted(actual_images.items())):
            findings.append("Los hashes de imágenes no coinciden con el lock.")
        split_counts = {
            split: sum(
                isinstance(sample, dict) and sample.get("split") == split for sample in samples
            )
            for split in ("train", "validation", "test")
        }
        if lock.get("split_counts") != split_counts:
            findings.append("Los splits no coinciden con el lock.")
    return DatasetReleaseVerification(not findings, actual_manifest_hash, tuple(findings))
