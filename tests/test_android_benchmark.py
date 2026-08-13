from pathlib import Path

import pytest

from avatar_face.domain.benchmarking import AndroidBenchmarkRequest


def test_android_benchmark_request_requires_explicit_serial() -> None:
    with pytest.raises(ValueError, match="serial"):
        AndroidBenchmarkRequest(" ", Path("app.apk"), Path("artifacts/android"))


def test_android_benchmark_request_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backends"):
        AndroidBenchmarkRequest(
            "ABC123",
            Path("app.apk"),
            Path("artifacts/android"),
            backends=("gpu",),
        )


def test_android_benchmark_request_limits_runs() -> None:
    with pytest.raises(ValueError, match="corridas"):
        AndroidBenchmarkRequest(
            "ABC123",
            Path("app.apk"),
            Path("artifacts/android"),
            runs=51,
        )


def test_android_benchmark_request_rejects_asset_path_traversal() -> None:
    with pytest.raises(ValueError, match="asset"):
        AndroidBenchmarkRequest(
            "ABC123",
            Path("app.apk"),
            Path("artifacts/android"),
            model_asset="../model.onnx",
        )
