from pathlib import Path

import pytest

from avatar_face.application.generate_smoke_dataset import GenerateSmokeDataset
from avatar_face.domain.dataset import DatasetSample
from avatar_face.infrastructure.dataset.json_auditor import JsonDatasetAuditor
from avatar_face.infrastructure.dataset.procedural_generator import (
    ProceduralAvatarDatasetGenerator,
)


def test_dataset_sample_rejects_non_synthetic_record() -> None:
    with pytest.raises(ValueError, match="sintéticas"):
        DatasetSample(
            "sample",
            "images/sample.png",
            "avatar",
            (),
            "unknown",
            "unknown",
            "CC0-1.0",
            "https://creativecommons.org/publicdomain/zero/1.0/",
            "unknown",
            "0" * 64,
            "train",
            synthetic=False,
        )


def test_procedural_dataset_is_deterministic_and_auditable(tmp_path: Path) -> None:
    generator = GenerateSmokeDataset(ProceduralAvatarDatasetGenerator())
    first = generator.execute(str(tmp_path / "first"), 10, 42)
    second = generator.execute(str(tmp_path / "second"), 10, 42)

    first_image = tmp_path / "first/images/avatar-00000.png"
    second_image = tmp_path / "second/images/avatar-00000.png"
    assert first.samples == 10
    assert first.train_samples == 8
    assert first.validation_samples == 1
    assert first.test_samples == 1
    assert first_image.read_bytes() == second_image.read_bytes()
    assert first.manifest_sha256 == second.manifest_sha256
    assert JsonDatasetAuditor().audit(Path(first.manifest_path)).approved


def test_dataset_audit_detects_modified_image(tmp_path: Path) -> None:
    result = GenerateSmokeDataset(ProceduralAvatarDatasetGenerator()).execute(
        str(tmp_path / "dataset"), 10, 7
    )
    image = tmp_path / "dataset/images/avatar-00000.png"
    image.write_bytes(image.read_bytes() + b"tampered")

    audit = JsonDatasetAuditor().audit(Path(result.manifest_path))

    assert not audit.approved
    assert any("Hash no coincide" in finding for finding in audit.findings)
