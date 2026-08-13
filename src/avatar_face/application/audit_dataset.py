from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from avatar_face.domain.dataset import DatasetAuditResult
from avatar_face.domain.ports import DatasetAuditor


@dataclass(frozen=True, slots=True)
class AuditDataset:
    """Caso de uso para ejecutar la compuerta legal y de integridad."""

    auditor: DatasetAuditor

    def execute(self, manifest_path: Path) -> DatasetAuditResult:
        return self.auditor.audit(manifest_path)
