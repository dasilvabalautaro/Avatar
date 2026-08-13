from __future__ import annotations

from avatar_face.domain.dataset import MicroTrainingConfig, MicroTrainingResult
from avatar_face.domain.ports import MicroTrainingRunner


class RunLocalMicroTraining:
    """Caso de uso del smoke training local antes de utilizar infraestructura remota."""

    def __init__(self, trainer: MicroTrainingRunner) -> None:
        self.trainer = trainer

    def execute(self, config: MicroTrainingConfig) -> MicroTrainingResult:
        return self.trainer.run(config)
