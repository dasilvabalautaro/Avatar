from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from avatar_face import __version__
from avatar_face.application.audit_candidates import AuditModelCandidates
from avatar_face.application.inspect_android import InspectAndroidEnvironment
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

    audit = commands.add_parser(
        "audit-candidates", help="Aplica la compuerta de licencias."
    )
    audit.add_argument(
        "--manifest", type=Path, default=Path("configs/model-candidates.json")
    )
    audit.add_argument("--json", action="store_true", dest="as_json")
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


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    if parsed.command == "status":
        return _status(parsed.adb_path, parsed.as_json)
    if parsed.command == "validate-prompt":
        return _validate_prompt(parsed.text, parsed.seed, parsed.image_size)
    if parsed.command == "audit-candidates":
        return _audit_candidates(parsed.manifest, parsed.as_json)
    raise AssertionError(f"Comando no implementado: {parsed.command}")


if __name__ == "__main__":
    raise SystemExit(main())
