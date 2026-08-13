from __future__ import annotations

import re
from dataclasses import dataclass

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DATASET_LICENSES = frozenset({"CC0-1.0", "CC-BY-4.0"})
DATASET_SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """Registro auditable de una imagen y sus derechos."""

    identifier: str
    image: str
    caption: str
    attributes: tuple[tuple[str, str], ...]
    source: str
    creator: str
    license_id: str
    license_url: str
    consent_or_release: str
    sha256: str
    split: str
    synthetic: bool

    def __post_init__(self) -> None:
        if not self.identifier or not self.caption.strip():
            raise ValueError("La muestra requiere id y caption.")
        if self.image.startswith("/") or ".." in self.image.split("/"):
            raise ValueError("La ruta de imagen debe ser relativa y segura.")
        if self.license_id not in DATASET_LICENSES:
            raise ValueError(f"Licencia de dataset no aprobada: {self.license_id}.")
        if self.split not in DATASET_SPLITS:
            raise ValueError(f"Split no válido: {self.split}.")
        if not SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError("SHA-256 inválido.")
        if not self.synthetic or self.consent_or_release != "not_applicable_synthetic":
            raise ValueError("El smoke dataset sólo admite muestras sintéticas.")


@dataclass(frozen=True, slots=True)
class DatasetGenerationResult:
    """Resumen verificable del dataset procedimental generado."""

    output_directory: str
    manifest_path: str
    samples: int
    train_samples: int
    validation_samples: int
    test_samples: int
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class DatasetAuditResult:
    """Resultado de verificar manifiesto, archivos y política legal."""

    manifest_path: str
    approved: bool
    samples: int
    unique_hashes: int
    findings: tuple[str, ...]
