from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from avatar_face.domain.ports import FeasibilityQuantizer
from avatar_face.domain.quantization import QuantizationResult


@dataclass(frozen=True, slots=True)
class QuantizeFeasibilityModel:
    """Cuantiza un ONNX sin acoplar el caso de uso al runtime elegido."""

    quantizer: FeasibilityQuantizer

    def execute(
        self,
        source_path: Path,
        output_directory: Path,
        overwrite: bool = False,
    ) -> QuantizationResult:
        return self.quantizer.quantize(source_path, output_directory, overwrite)
