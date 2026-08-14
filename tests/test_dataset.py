import hashlib
import json
from pathlib import Path

import pytest

from avatar_face.application.generate_smoke_dataset import GenerateSmokeDataset
from avatar_face.domain.dataset import DatasetSample
from avatar_face.infrastructure.dataset.json_auditor import JsonDatasetAuditor
from avatar_face.infrastructure.dataset.procedural_generator import (
    ProceduralAvatarDatasetGenerator,
)
from avatar_face.infrastructure.dataset.release_freezer import freeze_dataset, verify_frozen_dataset


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


def test_dataset_audit_detects_perceptually_identical_images(tmp_path: Path) -> None:
    result = GenerateSmokeDataset(ProceduralAvatarDatasetGenerator()).execute(
        str(tmp_path / "dataset"), 10, 7
    )
    manifest_path = Path(result.manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = manifest_path.parent / payload["samples"][0]["image"]
    second = manifest_path.parent / payload["samples"][1]["image"]
    second.write_bytes(first.read_bytes())
    payload["samples"][1]["sha256"] = hashlib.sha256(second.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    audit = JsonDatasetAuditor().audit(manifest_path)

    assert not audit.approved
    assert any("Similitud perceptual excesiva" in finding for finding in audit.findings)


def test_procedural_dataset_balances_primary_strata(tmp_path: Path) -> None:
    result = GenerateSmokeDataset(ProceduralAvatarDatasetGenerator()).execute(
        str(tmp_path / "dataset"), 64, 42
    )
    payload = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))

    for attribute in ("skin_tone", "hair_style", "expression", "accessory", "face_shape"):
        counts: dict[str, int] = {}
        for sample in payload["samples"]:
            value = sample["attributes"][attribute]
            counts[value] = counts.get(value, 0) + 1
        assert max(counts.values()) - min(counts.values()) <= 1


def test_expanded_dataset_is_auditable_and_can_be_frozen(tmp_path: Path) -> None:
    result = ProceduralAvatarDatasetGenerator().generate_training(
        str(tmp_path / "dataset"), 100, 42
    )
    manifest = Path(result.manifest_path)

    audit = JsonDatasetAuditor().audit(manifest)
    release = freeze_dataset(manifest, tmp_path / "dataset" / "dataset-v2.lock.json", "v2.0.0")

    assert audit.approved
    assert result.train_samples == 80
    assert result.validation_samples == 10
    assert result.test_samples == 10
    assert release.manifest_sha256 == result.manifest_sha256
    assert Path(release.lock_path).is_file()
    assert verify_frozen_dataset(manifest, Path(release.lock_path)).approved


def test_freeze_rejects_unapproved_dataset(tmp_path: Path) -> None:
    result = ProceduralAvatarDatasetGenerator().generate_training(
        str(tmp_path / "dataset"), 100, 42
    )
    manifest = Path(result.manifest_path)
    image = manifest.parent / "images/avatar-00000.png"
    image.write_bytes(image.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="auditoría debe aprobarse"):
        freeze_dataset(manifest, tmp_path / "dataset" / "dataset-v2.lock.json", "v2.0.0")
