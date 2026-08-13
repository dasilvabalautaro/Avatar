from pathlib import Path

import pytest
import torch

from avatar_face.application.generate_smoke_dataset import GenerateSmokeDataset
from avatar_face.domain.dataset import DatasetLoadConfig
from avatar_face.infrastructure.dataset.manifest_loader import ManifestTorchDataset
from avatar_face.infrastructure.dataset.procedural_generator import (
    ProceduralAvatarDatasetGenerator,
)


def _manifest(tmp_path: Path) -> Path:
    result = GenerateSmokeDataset(ProceduralAvatarDatasetGenerator()).execute(
        str(tmp_path / "dataset"), 20, 42
    )
    return Path(result.manifest_path)


def test_manifest_loader_returns_declared_normalized_train_batch(tmp_path: Path) -> None:
    loader = ManifestTorchDataset(DatasetLoadConfig(_manifest(tmp_path), batch_size=4, seed=12))

    batch = loader.batch()

    assert len(loader) == 16
    assert batch.images.shape == (4, 3, 256, 256)
    assert batch.images.dtype == torch.float32
    assert float(batch.images.min()) >= -1.0
    assert float(batch.images.max()) <= 1.0
    assert all(identifier.startswith("avatar-") for identifier in batch.identifiers)


def test_manifest_loader_order_is_deterministic_for_seed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    first = ManifestTorchDataset(DatasetLoadConfig(manifest, batch_size=5, seed=91)).batch()
    second = ManifestTorchDataset(DatasetLoadConfig(manifest, batch_size=5, seed=91)).batch()

    assert first.identifiers == second.identifiers
    assert torch.equal(first.images, second.images)


def test_manifest_loader_rejects_hash_changed_before_preflight(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    image = manifest.parent / "images/avatar-00000.png"
    image.write_bytes(image.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="Hash no coincide"):
        ManifestTorchDataset(DatasetLoadConfig(manifest))


def test_manifest_loader_rechecks_hash_when_opening_sample(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    loader = ManifestTorchDataset(DatasetLoadConfig(manifest, batch_size=20, seed=1))
    selected = loader.samples[0]
    image = manifest.parent / selected.image
    image.write_bytes(image.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="Hash no coincide al abrir"):
        loader.batch()
