from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from avatar_face import __version__
from avatar_face.application.audit_candidates import AuditModelCandidates
from avatar_face.application.audit_dataset import AuditDataset
from avatar_face.application.export_components import ExportFeasibilityComponents
from avatar_face.application.export_feasibility import ExportFeasibilityModel
from avatar_face.application.generate_smoke_dataset import GenerateSmokeDataset
from avatar_face.application.inspect_android import InspectAndroidEnvironment
from avatar_face.application.quantize_feasibility import QuantizeFeasibilityModel
from avatar_face.application.run_android_benchmark import RunAndroidBenchmark
from avatar_face.domain.benchmarking import AndroidBenchmarkRequest
from avatar_face.domain.feasibility import FEASIBILITY_PROFILES, get_feasibility_profile
from avatar_face.domain.licensing import PermissiveLicensePolicy
from avatar_face.domain.models import AvatarPrompt, InvalidPromptError
from avatar_face.infrastructure.android.adb_probe import AdbDeviceProbe
from avatar_face.infrastructure.model_candidate_repository import (
    JsonModelCandidateRepository,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="avatar-face",
        description="Herramientas reproducibles del proyecto AvatarFace.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="Inspecciona el entorno Android.")
    status.add_argument("--json", action="store_true", dest="as_json")
    status.add_argument("--adb-path", default="adb")

    validate = commands.add_parser("validate-prompt", help="Valida un prompt.")
    validate.add_argument("text")
    validate.add_argument("--seed", type=int, default=42)
    validate.add_argument("--image-size", type=int, default=256)

    audit = commands.add_parser("audit-candidates", help="Aplica la compuerta de licencias.")
    audit.add_argument("--manifest", type=Path, default=Path("configs/model-candidates.json"))
    audit.add_argument("--json", action="store_true", dest="as_json")

    feasibility = commands.add_parser(
        "describe-feasibility", help="Describe un perfil sintético móvil."
    )
    feasibility.add_argument("--profile", choices=tuple(FEASIBILITY_PROFILES), default="micro")
    feasibility.add_argument("--json", action="store_true", dest="as_json")

    export = commands.add_parser(
        "export-feasibility", help="Exporta y valida un perfil ONNX sintético."
    )
    export.add_argument("--profile", choices=tuple(FEASIBILITY_PROFILES), default="micro")
    export.add_argument("--output-dir", default="artifacts/feasibility")
    export.add_argument("--overwrite", action="store_true")

    components = commands.add_parser(
        "export-feasibility-components",
        help="Exporta encoder, denoiser y decoder para perfilado.",
    )
    components.add_argument("--profile", choices=tuple(FEASIBILITY_PROFILES), default="bridge")
    components.add_argument("--output-dir", default="artifacts/feasibility")
    components.add_argument("--overwrite", action="store_true")

    quantize = commands.add_parser(
        "quantize-feasibility", help="Cuantiza un modelo ONNX sintético a INT8 QDQ."
    )
    quantize.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/feasibility/avatarface-feasibility-micro.onnx"),
    )
    quantize.add_argument("--output-dir", type=Path, default=Path("artifacts/feasibility"))
    quantize.add_argument("--skip-preprocess", action="store_true")
    quantize.add_argument("--overwrite", action="store_true")

    benchmark = commands.add_parser(
        "benchmark-android", help="Mide el APK en un Android físico por USB."
    )
    benchmark.add_argument("--serial", required=True)
    benchmark.add_argument(
        "--apk", type=Path, default=Path("android/app/build/outputs/apk/debug/app-debug.apk")
    )
    benchmark.add_argument("--output-dir", type=Path, default=Path("artifacts/android"))
    benchmark.add_argument("--backend", choices=("cpu", "nnapi"), action="append", dest="backends")
    benchmark.add_argument("--runs", type=int, default=7)
    benchmark.add_argument("--model-asset", default="avatarface-feasibility-micro.onnx")
    benchmark.add_argument("--profile-operators", action="store_true")

    dataset = commands.add_parser(
        "generate-smoke-dataset",
        help="Genera avatares procedimentales CC0 y su manifiesto auditable.",
    )
    dataset.add_argument("--output-dir", default="data/smoke-procedural")
    dataset.add_argument("--samples", type=int, default=64)
    dataset.add_argument("--seed", type=int, default=42)
    dataset.add_argument("--overwrite", action="store_true")

    dataset_audit = commands.add_parser(
        "audit-dataset", help="Verifica licencias, hashes y duplicados del dataset."
    )
    dataset_audit.add_argument(
        "--manifest", type=Path, default=Path("data/smoke-procedural/manifest.json")
    )
    return parser


def _status(adb_path: str, as_json: bool) -> int:
    environment = InspectAndroidEnvironment(AdbDeviceProbe(adb_path)).execute()
    payload = asdict(environment)
    payload["ready"] = bool(environment.ready_devices) and environment.error is None
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"ADB: {environment.adb_path or 'no encontrado'}")
        print(f"Versión: {environment.adb_version or 'no disponible'}")
        print(f"Dispositivos listos: {len(environment.ready_devices)}")
        if environment.error:
            print(f"Error: {environment.error}")
    return 0 if environment.error is None else 1


def _validate_prompt(text: str, seed: int, image_size: int) -> int:
    try:
        prompt = AvatarPrompt(text, seed, image_size)
    except InvalidPromptError as error:
        print(f"Prompt inválido: {error}")
        return 2
    print(json.dumps(asdict(prompt), indent=2, ensure_ascii=False))
    return 0


def _audit_candidates(manifest: Path, as_json: bool) -> int:
    candidates = JsonModelCandidateRepository(manifest).load()
    audits = AuditModelCandidates(PermissiveLicensePolicy()).execute(candidates)
    by_identifier = {candidate.identifier: candidate for candidate in candidates}
    payload = []
    for audit in audits:
        candidate = by_identifier[audit.identifier]
        payload.append(
            {
                **asdict(audit),
                "android_fit_as_is": candidate.android_fit_as_is,
                "role": candidate.role,
            }
        )
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for record in payload:
            legal = "aprobado" if record["approved"] else "bloqueado"
            print(
                f"{record['identifier']}: licencia={legal}, "
                f"android={record['android_fit_as_is']}, rol={record['role']}"
            )
    return 0


def _describe_feasibility(profile_name: str, as_json: bool) -> int:
    profile = get_feasibility_profile(profile_name)
    payload = {
        **asdict(profile),
        "latent_tokens": profile.latent_tokens,
        "estimated_parameters": profile.estimated_parameters,
        "estimated_fp32_mib": profile.fp32_parameter_budget_bytes / 2**20,
        "estimated_int8_mib": profile.int8_parameter_budget_bytes / 2**20,
    }
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Perfil: {profile.name}")
        print(f"Parámetros estimados: {profile.estimated_parameters:,}")
        print(f"Pesos FP32 estimados: {payload['estimated_fp32_mib']:.1f} MiB")
        print(f"Pesos INT8 estimados: {payload['estimated_int8_mib']:.1f} MiB")
    return 0


def _export_feasibility(profile_name: str, output_dir: str, overwrite: bool) -> int:
    from avatar_face.infrastructure.feasibility.onnx_exporter import (
        OnnxFeasibilityExporter,
    )

    profile = get_feasibility_profile(profile_name)
    result = ExportFeasibilityModel(OnnxFeasibilityExporter()).execute(
        profile, output_dir, overwrite
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    return 0


def _export_components(profile_name: str, output_dir: str, overwrite: bool) -> int:
    from avatar_face.infrastructure.feasibility.onnx_exporter import (
        OnnxFeasibilityExporter,
    )

    profile = get_feasibility_profile(profile_name)
    results = ExportFeasibilityComponents(OnnxFeasibilityExporter()).execute(
        profile, output_dir, overwrite
    )
    print(json.dumps([asdict(result) for result in results], indent=2, default=str))
    return 0


def _benchmark_android(
    serial: str,
    apk: Path,
    output_directory: Path,
    backends: list[str] | None,
    runs: int,
    model_asset: str,
    profile_operators: bool,
) -> int:
    from avatar_face.infrastructure.android.adb_benchmark import AdbBenchmarkRunner

    request = AndroidBenchmarkRequest(
        serial=serial,
        apk_path=apk,
        output_directory=output_directory,
        backends=tuple(backends or ("cpu", "nnapi")),
        runs=runs,
        model_asset=model_asset,
        profile_operators=profile_operators,
    )
    results = RunAndroidBenchmark(AdbBenchmarkRunner()).execute(request)
    payload = [
        {
            "backend": result.backend,
            "output_path": str(result.output_path),
            "median_ms": result.payload.get("median_ms"),
        }
        for result in results
    ]
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _quantize_feasibility(
    source: Path,
    output_directory: Path,
    overwrite: bool,
    skip_preprocess: bool,
) -> int:
    from avatar_face.infrastructure.feasibility.onnx_quantizer import (
        OnnxStaticInt8Quantizer,
    )

    quantizer = OnnxStaticInt8Quantizer(preprocess=not skip_preprocess)
    result = QuantizeFeasibilityModel(quantizer).execute(source, output_directory, overwrite)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))
    return 0


def _generate_smoke_dataset(output_directory: str, samples: int, seed: int, overwrite: bool) -> int:
    from avatar_face.infrastructure.dataset.procedural_generator import (
        ProceduralAvatarDatasetGenerator,
    )

    result = GenerateSmokeDataset(ProceduralAvatarDatasetGenerator()).execute(
        output_directory, samples, seed, overwrite
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0


def _audit_dataset(manifest_path: Path) -> int:
    from avatar_face.infrastructure.dataset.json_auditor import JsonDatasetAuditor

    result = AuditDataset(JsonDatasetAuditor()).execute(manifest_path)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0 if result.approved else 3


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    if parsed.command == "status":
        return _status(parsed.adb_path, parsed.as_json)
    if parsed.command == "validate-prompt":
        return _validate_prompt(parsed.text, parsed.seed, parsed.image_size)
    if parsed.command == "audit-candidates":
        return _audit_candidates(parsed.manifest, parsed.as_json)
    if parsed.command == "describe-feasibility":
        return _describe_feasibility(parsed.profile, parsed.as_json)
    if parsed.command == "export-feasibility":
        return _export_feasibility(parsed.profile, parsed.output_dir, parsed.overwrite)
    if parsed.command == "export-feasibility-components":
        return _export_components(parsed.profile, parsed.output_dir, parsed.overwrite)
    if parsed.command == "quantize-feasibility":
        return _quantize_feasibility(
            parsed.source,
            parsed.output_dir,
            parsed.overwrite,
            parsed.skip_preprocess,
        )
    if parsed.command == "benchmark-android":
        return _benchmark_android(
            parsed.serial,
            parsed.apk,
            parsed.output_dir,
            parsed.backends,
            parsed.runs,
            parsed.model_asset,
            parsed.profile_operators,
        )
    if parsed.command == "generate-smoke-dataset":
        return _generate_smoke_dataset(
            parsed.output_dir, parsed.samples, parsed.seed, parsed.overwrite
        )
    if parsed.command == "audit-dataset":
        return _audit_dataset(parsed.manifest)
    raise AssertionError(f"Comando no implementado: {parsed.command}")


if __name__ == "__main__":
    raise SystemExit(main())
