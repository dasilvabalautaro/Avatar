from __future__ import annotations

from dataclasses import dataclass

from avatar_face.domain.exporting import FeasibilityExportResult
from avatar_face.domain.feasibility import FeasibilityProfile
from avatar_face.domain.ports import FeasibilityExporter


@dataclass(frozen=True, slots=True)
class ExportFeasibilityModel:
    """Caso de uso para producir un grafo móvil sintético verificable."""

    exporter: FeasibilityExporter

    def execute(
        self,
        profile: FeasibilityProfile,
        output_directory: str,
        overwrite: bool = False,
    ) -> FeasibilityExportResult:
        return self.exporter.export(profile, output_directory, overwrite)
