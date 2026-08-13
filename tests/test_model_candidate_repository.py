from pathlib import Path

import pytest

from avatar_face.infrastructure.model_candidate_repository import (
    JsonModelCandidateRepository,
)


def test_repository_loads_project_candidates() -> None:
    candidates = JsonModelCandidateRepository(
        Path("configs/model-candidates.json")
    ).load()

    assert len(candidates) == 9
    assert candidates[0].identifier.startswith("Efficient-Large-Model/")


def test_repository_rejects_unknown_schema(tmp_path: Path) -> None:
    manifest = tmp_path / "candidates.json"
    manifest.write_text('{"schema_version": 2, "candidates": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version=1"):
        JsonModelCandidateRepository(manifest).load()
