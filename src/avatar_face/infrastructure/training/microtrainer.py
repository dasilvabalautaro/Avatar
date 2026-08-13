from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from torch import Tensor, nn

from avatar_face.domain.dataset import DatasetLoadConfig, MicroTrainingConfig, MicroTrainingResult
from avatar_face.infrastructure.dataset.manifest_loader import ManifestTorchDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SmokeAutoencoder(nn.Module):
    """Autoencoder diminuto para validar datos, pérdida y checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=4),
            nn.SiLU(),
            nn.Conv2d(8, 8, kernel_size=4, stride=4),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=4),
            nn.SiLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=4),
            nn.Tanh(),
        )

    def forward(self, images: Tensor) -> Tensor:
        return cast(Tensor, self.decoder(self.encoder(images)))


class LocalMicroTrainer:
    """Entrena el smoke dataset en CPU y guarda evidencia suficiente para reanudar."""

    def run(self, config: MicroTrainingConfig) -> MicroTrainingResult:
        torch.manual_seed(config.seed)
        dataset = ManifestTorchDataset(config.dataset)
        if not len(dataset):
            raise ValueError(f"El split {config.dataset.split} no contiene muestras.")
        destination = config.output_directory.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint_path = destination / "microtraining-checkpoint.pt"
        reconstruction_path = destination / "validation-reconstruction.png"
        manifest_sha256 = _sha256(dataset.manifest_path)
        model = SmokeAutoencoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        start_step = 0
        if config.resume:
            start_step = self._restore(checkpoint_path, model, optimizer, manifest_sha256)
        if start_step > config.steps:
            raise ValueError("El checkpoint supera los steps solicitados.")

        model.train()
        loss = 0.0
        batches = (len(dataset) + config.dataset.batch_size - 1) // config.dataset.batch_size
        for step in range(start_step, config.steps):
            images = dataset.batch(step % batches).images
            reconstructed = model(images)
            step_loss = torch.nn.functional.mse_loss(reconstructed, images)
            optimizer.zero_grad(set_to_none=True)
            step_loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            loss = float(step_loss.detach())
        self._checkpoint(
            checkpoint_path, model, optimizer, config, manifest_sha256, config.steps, loss
        )
        self._save_reconstruction(reconstruction_path, model, config.dataset)
        return MicroTrainingResult(
            checkpoint_path,
            reconstruction_path,
            manifest_sha256,
            start_step,
            config.steps,
            loss,
        )

    def _restore(
        self,
        checkpoint_path: Path,
        model: SmokeAutoencoder,
        optimizer: torch.optim.Optimizer,
        manifest_sha256: str,
    ) -> int:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint inexistente: {checkpoint_path}")
        # El checkpoint es creado por este proceso y contiene la configuración
        # serializada (incluido ``Path``); declarar el modo evita que cambios de
        # PyTorch alteren el comportamiento de reanudación.
        payload: dict[str, Any] = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if payload.get("manifest_sha256") != manifest_sha256:
            raise ValueError("El checkpoint pertenece a otro manifiesto.")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng_state"])
        return int(payload["completed_steps"])

    def _checkpoint(
        self,
        checkpoint_path: Path,
        model: SmokeAutoencoder,
        optimizer: torch.optim.Optimizer,
        config: MicroTrainingConfig,
        manifest_sha256: str,
        completed_steps: int,
        final_loss: float,
    ) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "completed_steps": completed_steps,
                "seed": config.seed,
                "config": {
                    "dataset": {
                        "manifest_path": str(config.dataset.manifest_path),
                        "split": config.dataset.split,
                        "batch_size": config.dataset.batch_size,
                        "seed": config.dataset.seed,
                        "verify_hashes_on_read": config.dataset.verify_hashes_on_read,
                    },
                    "output_directory": str(config.output_directory),
                    "steps": config.steps,
                    "learning_rate": config.learning_rate,
                    "seed": config.seed,
                    "resume": config.resume,
                },
                "manifest_sha256": manifest_sha256,
                "final_loss": final_loss,
                "torch_rng_state": torch.get_rng_state(),
            },
            checkpoint_path,
        )

    def _save_reconstruction(
        self,
        reconstruction_path: Path,
        model: SmokeAutoencoder,
        dataset_config: DatasetLoadConfig,
    ) -> None:
        validation = ManifestTorchDataset(
            DatasetLoadConfig(
                manifest_path=dataset_config.manifest_path,
                split="validation",
                batch_size=1,
                seed=dataset_config.seed,
                verify_hashes_on_read=dataset_config.verify_hashes_on_read,
            )
        )
        model.eval()
        with torch.no_grad():
            batch = validation.batch()
            reconstruction = model(batch.images)
        image = reconstruction[0].add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
        rgb = image.permute(1, 2, 0).contiguous().cpu()
        Image.frombytes("RGB", (rgb.shape[1], rgb.shape[0]), bytes(rgb.flatten().tolist())).save(
            reconstruction_path
        )
