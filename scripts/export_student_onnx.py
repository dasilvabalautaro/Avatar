#!/usr/bin/env python3
"""Exporta el U-Net del estudiante (ADR 0010) a ONNX para el runtime móvil.

Se exporta **sólo la red**, no el bucle de muestreo: la app Android ejecuta los
8 pasos DDIM sobre el grafo, igual que el resto del pipeline móvil del
proyecto. Entradas: `sample` (b,3,256,256), `ratio` (b,) y `attributes`
(b, n_atributos, int64); salida: la predicción v.

Los pesos exportados son los **EMA**, que son los que generaron las muestras
de control evaluadas en la compuerta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from avatar_face.infrastructure.training.student_unet import StudentUNet, StudentUNetConfig


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", choices=("ema", "student"), default="ema")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    raw = state["student_config"]
    config = StudentUNetConfig(
        image_size=raw["image_size"],
        base_channels=raw["base_channels"],
        channel_multipliers=tuple(raw["channel_multipliers"]),
        residual_blocks_per_level=raw["residual_blocks_per_level"],
        attention_resolutions=tuple(raw["attention_resolutions"]),
        attention_heads=raw["attention_heads"],
        condition_dim=raw["condition_dim"],
        attribute_embedding_dim=raw["attribute_embedding_dim"],
        attribute_cardinalities=tuple(raw["attribute_cardinalities"]),
    )
    model = StudentUNet(config)
    model.load_state_dict(state[args.weights])
    model.eval().requires_grad_(False)

    size = config.image_size
    attributes_count = len(config.attribute_cardinalities)
    sample = torch.randn(1, 3, size, size)
    ratio = torch.full((1,), 0.5)
    attributes = torch.zeros(1, attributes_count, dtype=torch.long)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample, ratio, attributes),
        str(args.output),
        input_names=["sample", "ratio", "attributes"],
        output_names=["v_prediction"],
        opset_version=args.opset,
        dynamo=False,
        dynamic_axes={
            "sample": {0: "batch"},
            "ratio": {0: "batch"},
            "attributes": {0: "batch"},
            "v_prediction": {0: "batch"},
        },
    )

    parameters = model.parameter_count()
    metadata = {
        "schema_version": 1,
        "source_checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_of(args.checkpoint),
        "weights": args.weights,
        "completed_steps": state["completed_steps"],
        "formulation": state["train_config"]["formulation"],
        "ddim_steps": state["train_config"]["ddim_steps"],
        "student_config": raw,
        "parameters": parameters,
        "fp32_parameter_mib": parameters * 4 / 2**20,
        "int8_parameter_mib": parameters / 2**20,
        "onnx_path": str(args.output),
        "onnx_sha256": sha256_of(args.output),
        "onnx_bytes": args.output.stat().st_size,
        "opset": args.opset,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
