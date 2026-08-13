from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class QuantizationResult:
    """Evidencia de una cuantización ONNX reproducible."""

    source_path: Path
    model_path: Path
    manifest_path: Path
    source_bytes: int
    model_bytes: int
    compression_ratio: float
    sha256: str
    maximum_absolute_error: float
    mean_absolute_error: float
    quantization_seconds: float
