from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FeasibilityExportResult:
    """Evidencia producida al exportar y comparar un perfil sintético."""

    profile: str
    model_path: Path
    manifest_path: Path
    parameters: int
    model_bytes: int
    sha256: str
    maximum_absolute_error: float
    export_seconds: float


@dataclass(frozen=True, slots=True)
class ComponentExportResult:
    """Evidencia de un componente ONNX aislado para perfilado."""

    profile: str
    component: str
    model_path: Path
    manifest_path: Path
    parameters: int
    model_bytes: int
    sha256: str
    maximum_absolute_error: float
