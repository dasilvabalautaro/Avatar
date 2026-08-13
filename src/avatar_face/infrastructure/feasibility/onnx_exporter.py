from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import torch

from avatar_face.domain.exporting import ComponentExportResult, FeasibilityExportResult
from avatar_face.domain.feasibility import FeasibilityProfile
from avatar_face.infrastructure.feasibility.torch_model import (
    AvatarFaceFeasibilityModel,
    component_models,
    count_parameters,
    example_inputs,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class OnnxFeasibilityExporter:
    """Exporta el grafo y comprueba su equivalencia con ONNX Runtime CPU."""

    opset_version: int = 17
    seed: int = 42

    def export(
        self,
        profile: FeasibilityProfile,
        output_directory: str,
        overwrite: bool = False,
    ) -> FeasibilityExportResult:
        destination = Path(output_directory).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        model_path = destination / f"avatarface-feasibility-{profile.name}.onnx"
        manifest_path = destination / f"avatarface-feasibility-{profile.name}.json"
        if not overwrite and (model_path.exists() or manifest_path.exists()):
            raise FileExistsError(f"El artefacto ya existe para {profile.name}; usa --overwrite.")

        torch.manual_seed(self.seed)
        model = AvatarFaceFeasibilityModel(profile).eval()
        inputs = example_inputs(profile)
        with torch.inference_mode():
            expected = model(*inputs).numpy()

        started = time.perf_counter()
        torch.onnx.export(
            model,
            inputs,
            model_path,
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=["token_ids", "latent"],
            output_names=["image"],
        )
        export_seconds = time.perf_counter() - started

        graph = onnx.load(model_path)
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        actual = session.run(
            ["image"],
            {"token_ids": inputs[0].numpy(), "latent": inputs[1].numpy()},
        )[0]
        maximum_absolute_error = float(np.max(np.abs(expected - actual)))
        parameters = count_parameters(model)
        digest = _sha256(model_path)
        payload = {
            "schema_version": 1,
            "profile": asdict(profile),
            "parameters": parameters,
            "estimated_parameters": profile.estimated_parameters,
            "model_bytes": model_path.stat().st_size,
            "sha256": digest,
            "opset_version": self.opset_version,
            "seed": self.seed,
            "maximum_absolute_error": maximum_absolute_error,
            "export_seconds": export_seconds,
            "versions": {
                "torch": torch.__version__,
                "onnx": onnx.__version__,
                "onnxruntime": ort.__version__,
                "numpy": np.__version__,
            },
            "providers": session.get_providers(),
            "quality_warning": "Pesos aleatorios; este artefacto no genera avatares útiles.",
        }
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return FeasibilityExportResult(
            profile=profile.name,
            model_path=model_path,
            manifest_path=manifest_path,
            parameters=parameters,
            model_bytes=model_path.stat().st_size,
            sha256=digest,
            maximum_absolute_error=maximum_absolute_error,
            export_seconds=export_seconds,
        )

    def export_components(
        self,
        profile: FeasibilityProfile,
        output_directory: str,
        overwrite: bool = False,
    ) -> tuple[ComponentExportResult, ...]:
        destination = Path(output_directory).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self.seed)
        full_model = AvatarFaceFeasibilityModel(profile).eval()
        results = []

        for component, spec in component_models(full_model).items():
            model, inputs, input_names, output_name = spec
            model_path = destination / f"avatarface-feasibility-{profile.name}-{component}.onnx"
            manifest_path = destination / f"avatarface-feasibility-{profile.name}-{component}.json"
            if not overwrite and (model_path.exists() or manifest_path.exists()):
                raise FileExistsError(f"El componente {component} ya existe; usa --overwrite.")

            with torch.inference_mode():
                expected = model(*inputs).numpy()
            started = time.perf_counter()
            torch.onnx.export(
                model,
                inputs,
                model_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=list(input_names),
                output_names=[output_name],
            )
            export_seconds = time.perf_counter() - started
            onnx.checker.check_model(onnx.load(model_path))
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            runtime_inputs = {
                name: value.numpy() for name, value in zip(input_names, inputs, strict=True)
            }
            actual = session.run([output_name], runtime_inputs)[0]
            error = float(np.max(np.abs(expected - actual)))
            digest = _sha256(model_path)
            parameters = count_parameters(model)
            payload = {
                "schema_version": 1,
                "profile": profile.name,
                "component": component,
                "parameters": parameters,
                "model_bytes": model_path.stat().st_size,
                "sha256": digest,
                "input_names": input_names,
                "output_name": output_name,
                "maximum_absolute_error": error,
                "export_seconds": export_seconds,
                "seed": self.seed,
                "opset_version": self.opset_version,
                "quality_warning": "Pesos aleatorios; sólo sirve para perfilado técnico.",
            }
            manifest_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            results.append(
                ComponentExportResult(
                    profile.name,
                    component,
                    model_path,
                    manifest_path,
                    parameters,
                    model_path.stat().st_size,
                    digest,
                    error,
                )
            )
        return tuple(results)
