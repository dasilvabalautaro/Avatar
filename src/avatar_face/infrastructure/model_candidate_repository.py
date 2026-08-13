from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from avatar_face.domain.licensing import ModelCandidate, ModelComponent


@dataclass(frozen=True, slots=True)
class JsonModelCandidateRepository:
    """Carga candidatos desde un manifiesto JSON versionable."""

    path: Path

    def load(self) -> tuple[ModelCandidate, ...]:
        payload: Any = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("El manifiesto debe usar schema_version=1.")
        records = payload.get("candidates")
        if not isinstance(records, list):
            raise ValueError("candidates debe ser una lista.")
        return tuple(self._candidate(record) for record in records)

    @staticmethod
    def _candidate(record: Any) -> ModelCandidate:
        if not isinstance(record, dict):
            raise ValueError("Cada candidato debe ser un objeto.")
        components_payload = record.get("components")
        if not isinstance(components_payload, list) or not components_payload:
            raise ValueError("Cada candidato debe declarar componentes.")
        components = tuple(
            ModelComponent(
                name=str(component["name"]),
                license_id=str(component["license_id"]),
                source_url=str(component["source_url"]),
                has_use_restrictions=bool(component.get("has_use_restrictions", False)),
                notes=str(component.get("notes", "")),
            )
            for component in components_payload
            if isinstance(component, dict)
        )
        if len(components) != len(components_payload):
            raise ValueError("Todos los componentes deben ser objetos.")
        revision = record.get("revision")
        size = record.get("estimated_download_gib")
        return ModelCandidate(
            identifier=str(record["identifier"]),
            revision=None if revision is None else str(revision),
            estimated_download_gib=None if size is None else float(size),
            components=components,
            android_fit_as_is=bool(record.get("android_fit_as_is", False)),
            role=str(record.get("role", "discarded")),
        )
