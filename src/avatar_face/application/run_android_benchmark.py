from __future__ import annotations

from dataclasses import dataclass

from avatar_face.domain.benchmarking import AndroidBenchmarkRequest, AndroidBenchmarkResult
from avatar_face.domain.ports import AndroidBenchmarkRunner


@dataclass(frozen=True, slots=True)
class RunAndroidBenchmark:
    """Orquesta un benchmark sin conocer los detalles de ADB."""

    runner: AndroidBenchmarkRunner

    def execute(self, request: AndroidBenchmarkRequest) -> tuple[AndroidBenchmarkResult, ...]:
        return self.runner.run(request)
