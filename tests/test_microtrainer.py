from pathlib import Path

import torch
from PIL import Image

from avatar_face.application.generate_smoke_dataset import GenerateSmokeDataset
from avatar_face.domain.dataset import DatasetLoadConfig, MicroTrainingConfig
from avatar_face.infrastructure.dataset.procedural_generator import (
    ProceduralAvatarDatasetGenerator,
)
from avatar_face.infrastructure.training.microtrainer import LocalMicroTrainer


def test_microtrainer_saves_checkpoint_and_resumes(tmp_path: Path) -> None:
    generated = GenerateSmokeDataset(ProceduralAvatarDatasetGenerator()).execute(
        str(tmp_path / "dataset"), 10, 42
    )
    dataset = DatasetLoadConfig(Path(generated.manifest_path), batch_size=2, seed=12)
    output = tmp_path / "artifacts"
    trainer = LocalMicroTrainer()

    first = trainer.run(MicroTrainingConfig(dataset, output, steps=2, seed=20))
    resumed = trainer.run(MicroTrainingConfig(dataset, output, steps=4, seed=20, resume=True))

    checkpoint = torch.load(resumed.checkpoint_path, map_location="cpu")
    assert first.start_step == 0
    assert first.completed_steps == 2
    assert resumed.start_step == 2
    assert resumed.completed_steps == 4
    assert checkpoint["manifest_sha256"] == generated.manifest_sha256
    assert checkpoint["completed_steps"] == 4
    with Image.open(resumed.reconstruction_path) as reconstruction:
        assert reconstruction.size == (256, 256)
