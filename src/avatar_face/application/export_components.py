from __future__ import annotations

from dataclasses import dataclass

from avatar_face.domain.exporting import ComponentExportResult
from avatar_face.domain.feasibility import FeasibilityProfile
from avatar_face.domain.ports import FeasibilityComponentExporter


@dataclass(frozen=True, slots=True)
class ExportFeasibilityComponents:
    """Caso de uso para producir componentes ONNX perfilables."""

    exporter: FeasibilityComponentExporter

    def execute(
        self,
        profile: FeasibilityProfile,
        output_directory: str,
        overwrite: bool = False,
    ) -> tuple[ComponentExportResult, ...]:
        return self.exporter.export_components(profile, output_directory, overwrite)
