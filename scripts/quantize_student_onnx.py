#!/usr/bin/env python3
"""Cuantiza a INT8 QDQ el U-Net del estudiante (ADR 0006 y 0010).

El calibrador genérico del proyecto no sirve aquí: la entrada `attributes` son
índices de un vocabulario cerrado y los datos aleatorios caen fuera de rango.
Este script calibra con **datos representativos reales**: atributos válidos de
la release de destilación y estados x_t generados con el mismo schedule coseno
y los mismos ratios de la rejilla DDIM de 8 pasos que usa la inferencia.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

from avatar_face.domain.attributes import ATTRIBUTE_ORDER, AvatarAttributes

DDIM_STEPS = 8


def cosine_alpha_bar(t: float) -> float:
    s = 0.008
    return min(max(math.cos((t + s) / (1 + s) * math.pi / 2) ** 2, 1e-4), 0.9999)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StudentCalibrationReader(CalibrationDataReader):
    """Estados (x_t, ratio, atributos) representativos de la cadena de muestreo."""

    def __init__(self, dataset_dir: Path, samples: int, seed: int) -> None:
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        records = [r for r in manifest["samples"] if r["split"] == "train"][:samples]
        if not records:
            raise SystemExit("El manifiesto no tiene muestras del split train.")
        rng = np.random.default_rng(seed)
        grid = np.linspace(1.0, 0.0, DDIM_STEPS + 1)[:DDIM_STEPS]
        self._data: list[dict[str, np.ndarray]] = []
        for index, record in enumerate(records):
            with Image.open(dataset_dir / record["image"]) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
            x0 = (pixels / 127.5 - 1.0).transpose(2, 0, 1)[None]
            attributes = AvatarAttributes(
                **{k: record["attributes"][k] for k in ATTRIBUTE_ORDER}
            ).indices()
            t = float(grid[index % DDIM_STEPS])
            ab = cosine_alpha_bar(t)
            noise = rng.standard_normal(x0.shape).astype(np.float32)
            x_t = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * noise
            self._data.append(
                {
                    "sample": x_t.astype(np.float32),
                    "ratio": np.array([t], dtype=np.float32),
                    "attributes": np.array([attributes], dtype=np.int64),
                }
            )
        self._iterator = iter(self._data)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iterator, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/distill-teacher-v1"))
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-fp32",
        default="",
        help="patrones de nombre de nodo que NO se cuantizan, separados por comas "
        "(cuantización selectiva del ADR 0006; la vía de condicionamiento pesa poco "
        "en cómputo y mucho en precisión)",
    )
    parser.add_argument(
        "--calibrate-method",
        choices=("minmax", "entropy", "percentile"),
        default="minmax",
        help="MinMax deja que un valor atípico fije la escala; percentil y entropía "
        "la recortan y conservan resolución en el rango útil",
    )
    parser.add_argument(
        "--activation-bits",
        type=int,
        choices=(8, 16),
        default=8,
        help="16 bits en activaciones conserva precisión en difusores pequeños, "
        "a costa de kernels menos optimizados",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} ya existe; usa --overwrite.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preprocessed = args.output.with_name(args.output.stem + "-preprocessed.onnx")
    quant_pre_process(input_model=args.source, output_model_path=preprocessed)
    patterns = [p.strip() for p in args.keep_fp32.split(",") if p.strip()]
    excluded: list[str] = []
    if patterns:
        import onnx

        graph = onnx.load(str(preprocessed)).graph
        excluded = [
            node.name
            for node in graph.node
            if node.name and any(pattern in node.name for pattern in patterns)
        ]
        print(f"nodos en fp32: {len(excluded)} de {len(graph.node)}")
    quantize_static(
        nodes_to_exclude=excluded,
        model_input=preprocessed,
        model_output=args.output,
        calibration_data_reader=StudentCalibrationReader(
            args.dataset_dir, args.calibration_samples, args.seed
        ),
        quant_format=QuantFormat.QDQ,
        activation_type=(
            QuantType.QUInt16 if args.activation_bits == 16 else QuantType.QUInt8
        ),
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method={
            "minmax": CalibrationMethod.MinMax,
            "entropy": CalibrationMethod.Entropy,
            "percentile": CalibrationMethod.Percentile,
        }[args.calibrate_method],
    )
    preprocessed.unlink(missing_ok=True)
    payload = {
        "source": str(args.source),
        "source_sha256": sha256_of(args.source),
        "source_bytes": args.source.stat().st_size,
        "output": str(args.output),
        "output_sha256": sha256_of(args.output),
        "output_bytes": args.output.stat().st_size,
        "calibration_samples": args.calibration_samples,
        "quant_format": "QDQ",
        "fp32_nodes": len(excluded),
        "keep_fp32_patterns": patterns,
        "per_channel": True,
        "calibrate_method": args.calibrate_method,
        "activation_bits": args.activation_bits,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
