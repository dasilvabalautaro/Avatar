from __future__ import annotations

from dataclasses import dataclass

from avatar_face.domain.licensing import (
    CandidateAudit,
    ModelCandidate,
    PermissiveLicensePolicy,
)


@dataclass(frozen=True, slots=True)
class AuditModelCandidates:
    """Aplica la misma política a una colección de candidatos."""

    policy: PermissiveLicensePolicy

    def execute(self, candidates: tuple[ModelCandidate, ...]) -> tuple[CandidateAudit, ...]:
        return tuple(self.policy.audit(candidate) for candidate in candidates)
