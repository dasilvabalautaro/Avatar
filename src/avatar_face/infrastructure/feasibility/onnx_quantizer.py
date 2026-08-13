from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
from onnxruntime.quantization import (  # type: ignore[import-untyped]
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import (  # type: ignore[import-untyped]
    quant_pre_process,
)

from avatar_face.domain.quantization import QuantizationResult


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _representative_inputs(
    inputs: list[Any], random: np.random.Generator
) -> dict[str, np.ndarray[Any, Any]]:
    values = {}
    for metadata in inputs:
        shape = tuple(
            dimension if isinstance(dimension, int) else 1 for dimension in metadata.shape
        )
        if metadata.type == "tensor(int64)":
            values[metadata.name] = random.integers(0, 128, size=shape, dtype=np.int64)
        else:
            values[metadata.name] = random.standard_normal(shape).astype(np.float32)
    return values


class _RepresentativeDataReader(CalibrationDataReader):  # type: ignore[misc]
    def __init__(self, model_path: Path, samples: int, seed: int) -> None:
        random = np.random.default_rng(seed)
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        self._iterator = iter(_representative_inputs(inputs, random) for _ in range(samples))

    def get_next(self) -> dict[str, Any] | None:
        return next(self._iterator, None)


@dataclass(frozen=True, slots=True)
class OnnxStaticInt8Quantizer:
    """Genera QDQ INT8 con datos sintéticos reproducibles de calibración."""

    calibration_samples: int = 16
    validation_samples: int = 4
    seed: int = 42
    preprocess: bool = True

    def quantize(
        self,
        source_path: Path,
        output_directory: Path,
        overwrite: bool = False,
    ) -> QuantizationResult:
        source = source_path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Modelo ONNX no encontrado: {source}")
        destination = output_directory.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        suffix = "int8-preprocessed" if self.preprocess else "int8"
        model_path = destination / f"{source.stem}-{suffix}.onnx"
        manifest_path = destination / f"{source.stem}-{suffix}.json"
        if not overwrite and (model_path.exists() or manifest_path.exists()):
            raise FileExistsError("El artefacto INT8 ya existe; usa --overwrite.")

        started = time.perf_counter()
        if self.preprocess:
            with TemporaryDirectory(prefix="avatarface-quant-") as temporary_directory:
                preprocessed_path = Path(temporary_directory) / "preprocessed.onnx"
                quant_pre_process(input_model=source, output_model_path=preprocessed_path)
                self._quantize_onnx(preprocessed_path, model_path)
        else:
            self._quantize_onnx(source, model_path)
        quantization_seconds = time.perf_counter() - started
        onnx.checker.check_model(onnx.load(model_path))

        maximum_error, mean_error = self._compare(source, model_path)
        source_bytes = source.stat().st_size
        model_bytes = model_path.stat().st_size
        digest = _sha256(model_path)
        payload = {
            "schema_version": 1,
            "method": "static_qdq_int8",
            "preprocessed": self.preprocess,
            "source_path": str(source),
            "source_bytes": source_bytes,
            "model_path": str(model_path),
            "model_bytes": model_bytes,
            "compression_ratio": source_bytes / model_bytes,
            "sha256": digest,
            "calibration_samples": self.calibration_samples,
            "validation_samples": self.validation_samples,
            "seed": self.seed,
            "activation_type": "QUInt8",
            "weight_type": "QInt8",
            "per_channel": True,
            "quantized_operators": ["Conv", "Gemm", "MatMul"],
            "maximum_absolute_error": maximum_error,
            "mean_absolute_error": mean_error,
            "quantization_seconds": quantization_seconds,
            "onnxruntime": ort.__version__,
            "quality_warning": "Calibración sintética y pesos aleatorios; no mide calidad visual.",
        }
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return QuantizationResult(
            source,
            model_path,
            manifest_path,
            source_bytes,
            model_bytes,
            source_bytes / model_bytes,
            digest,
            maximum_error,
            mean_error,
            quantization_seconds,
        )

    def _quantize_onnx(self, source: Path, destination: Path) -> None:
        quantize_static(
            model_input=str(source),
            model_output=str(destination),
            calibration_data_reader=_RepresentativeDataReader(
                source, self.calibration_samples, self.seed
            ),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=["Conv", "Gemm", "MatMul"],
        )

    def _compare(self, source: Path, quantized: Path) -> tuple[float, float]:
        reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
        candidate = ort.InferenceSession(str(quantized), providers=["CPUExecutionProvider"])
        random = np.random.default_rng(self.seed + 1)
        absolute_errors = []
        for _ in range(self.validation_samples):
            inputs = _representative_inputs(reference.get_inputs(), random)
            output_name = reference.get_outputs()[0].name
            expected = reference.run([output_name], inputs)[0]
            actual = candidate.run([output_name], inputs)[0]
            absolute_errors.append(np.abs(expected - actual))
        combined = np.concatenate(tuple(error.ravel() for error in absolute_errors))
        return float(np.max(combined)), float(np.mean(combined))
