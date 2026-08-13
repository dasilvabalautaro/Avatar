from __future__ import annotations

from pathlib import Path
from typing import Protocol

from avatar_face.domain.benchmarking import AndroidBenchmarkRequest, AndroidBenchmarkResult
from avatar_face.domain.dataset import DatasetAuditResult, DatasetGenerationResult
from avatar_face.domain.exporting import ComponentExportResult, FeasibilityExportResult
from avatar_face.domain.feasibility import FeasibilityProfile
from avatar_face.domain.models import AndroidEnvironment
from avatar_face.domain.quantization import QuantizationResult


class AndroidDeviceProbe(Protocol):
    """Puerto para inspeccionar Android sin acoplar la aplicación a ADB."""

    def inspect(self) -> AndroidEnvironment:
        """Devuelve el estado observable del entorno Android."""
        ...


class SmokeDatasetGenerator(Protocol):
    """Puerto para generar muestras propias y auditables."""

    def generate(
        self,
        output_directory: str,
        samples: int,
        seed: int,
        overwrite: bool = False,
    ) -> DatasetGenerationResult:
        """Genera imágenes y manifiesto, sin descargar activos externos."""
        ...


class DatasetAuditor(Protocol):
    """Puerto para verificar un dataset sin acoplar la aplicación a JSON."""

    def audit(self, manifest_path: Path) -> DatasetAuditResult:
        """Recalcula evidencia y aplica la política legal del dataset."""
        ...


class AndroidBenchmarkRunner(Protocol):
    """Puerto para medir el APK en un dispositivo Android concreto."""

    def run(self, request: AndroidBenchmarkRequest) -> tuple[AndroidBenchmarkResult, ...]:
        """Instala, ejecuta y persiste los resultados del benchmark."""
        ...


class FeasibilityExporter(Protocol):
    """Puerto para exportar un perfil sin acoplar la aplicación a ONNX."""

    def export(
        self,
        profile: FeasibilityProfile,
        output_directory: str,
        overwrite: bool = False,
    ) -> FeasibilityExportResult:
        """Exporta, valida y devuelve la evidencia del artefacto."""
        ...


class FeasibilityComponentExporter(Protocol):
    """Puerto para exportar componentes aislados del grafo sintético."""

    def export_components(
        self,
        profile: FeasibilityProfile,
        output_directory: str,
        overwrite: bool = False,
    ) -> tuple[ComponentExportResult, ...]:
        """Exporta encoder, denoiser y decoder con pesos reproducibles."""
        ...


class FeasibilityQuantizer(Protocol):
    """Puerto para cuantización móvil sin acoplar la aplicación a ONNX Runtime."""

    def quantize(
        self,
        source_path: Path,
        output_directory: Path,
        overwrite: bool = False,
    ) -> QuantizationResult:
        """Cuantiza y compara el modelo contra su referencia FP32."""
        ...
