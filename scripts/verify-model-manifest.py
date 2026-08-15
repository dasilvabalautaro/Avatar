#!/usr/bin/env python3
"""Verifica integridad y procedencia del paquete local de pesos congelado."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    components = payload["components"]
    if payload.get("schema_version") != 1 or not isinstance(components, list) or not components:
        raise SystemExit("ERROR: manifiesto de modelo inválido")

    verified = 0
    total_bytes = 0
    for component in components:
        for record in component["files"]:
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise SystemExit(f"ERROR: ruta insegura: {relative}")
            if not relative.parts or relative.parts[0] != component["path"]:
                raise SystemExit(f"ERROR: ruta fuera del componente: {relative}")
            path = manifest.parent / relative
            if not path.is_file():
                raise SystemExit(f"ERROR: archivo faltante: {path}")
            if path.stat().st_size != record["bytes"]:
                raise SystemExit(f"ERROR: tamaño inválido: {path}")
            if sha256_file(path) != record["sha256"]:
                raise SystemExit(f"ERROR: SHA-256 inválido: {path}")
            verified += 1
            total_bytes += record["bytes"]
    print(
        "model_manifest_ok "
        f"components={len(components)} files={verified} bytes={total_bytes} "
        f"manifest_sha256={sha256_file(manifest)}"
    )


if __name__ == "__main__":
    main()
