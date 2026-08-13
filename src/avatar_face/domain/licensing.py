from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelComponent:
    """Componente auditable de un pipeline generativo."""

    name: str
    license_id: str
    source_url: str
    has_use_restrictions: bool = False
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    """Candidato completo, no sólo su denoiser principal."""

    identifier: str
    revision: str | None
    estimated_download_gib: float | None
    components: tuple[ModelComponent, ...]
    android_fit_as_is: bool
    role: str


@dataclass(frozen=True, slots=True)
class LicenseFinding:
    """Hallazgo que impide aprobar automáticamente un candidato."""

    component: str
    reason: str


@dataclass(frozen=True, slots=True)
class CandidateAudit:
    """Resultado determinista de aplicar la política de licencias."""

    identifier: str
    approved: bool
    findings: tuple[LicenseFinding, ...]


@dataclass(frozen=True, slots=True)
class PermissiveLicensePolicy:
    """Política estricta para componentes redistribuibles de AvatarFace."""

    allowed_license_ids: frozenset[str] = frozenset(
        {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0"}
    )

    def audit(self, candidate: ModelCandidate) -> CandidateAudit:
        findings: list[LicenseFinding] = []
        for component in candidate.components:
            if component.license_id not in self.allowed_license_ids:
                findings.append(
                    LicenseFinding(
                        component.name,
                        f"Licencia no aprobada: {component.license_id}.",
                    )
                )
            if component.has_use_restrictions:
                findings.append(
                    LicenseFinding(component.name, "Contiene restricciones de uso.")
                )
        if candidate.revision is None:
            findings.append(
                LicenseFinding("candidate", "Falta fijar una revisión inmutable.")
            )
        return CandidateAudit(candidate.identifier, not findings, tuple(findings))
