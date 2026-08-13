from __future__ import annotations

from dataclasses import dataclass

from avatar_face.domain.dataset import DatasetGenerationResult
from avatar_face.domain.ports import SmokeDatasetGenerator


@dataclass(frozen=True, slots=True)
class GenerateSmokeDataset:
    """Genera un dataset sintético sin acoplar el caso de uso a Pillow."""

    generator: SmokeDatasetGenerator

    def execute(
        self,
        output_directory: str,
        samples: int,
        seed: int,
        overwrite: bool = False,
    ) -> DatasetGenerationResult:
        if not 10 <= samples <= 10_000:
            raise ValueError("El smoke dataset debe contener entre 10 y 10,000 muestras.")
        return self.generator.generate(output_directory, samples, seed, overwrite)
