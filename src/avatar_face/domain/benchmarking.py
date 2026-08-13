from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AndroidBenchmarkRequest:
    """Parámetros validados para un benchmark sobre un Android físico."""

    serial: str
    apk_path: Path
    output_directory: Path
    backends: tuple[str, ...] = ("cpu", "nnapi")
    runs: int = 7
    model_asset: str = "avatarface-feasibility-micro.onnx"
    profile_operators: bool = False

    def __post_init__(self) -> None:
        if not self.serial.strip():
            raise ValueError("El serial ADB es obligatorio.")
        if not self.backends or any(item not in {"cpu", "nnapi"} for item in self.backends):
            raise ValueError("Los backends válidos son cpu y nnapi.")
        if not 1 <= self.runs <= 50:
            raise ValueError("Las corridas deben estar entre 1 y 50.")
        if "/" in self.model_asset or not self.model_asset.endswith(".onnx"):
            raise ValueError("El modelo debe ser un asset ONNX sin directorios.")


@dataclass(frozen=True, slots=True)
class AndroidBenchmarkResult:
    """Evidencia persistida de una ejecución del benchmark Android."""

    backend: str
    output_path: Path
    payload: dict[str, Any]
