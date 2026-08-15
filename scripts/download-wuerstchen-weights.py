#!/usr/bin/env python3
"""Descarga los pesos fijados por el manifiesto y verifica su SHA-256.

Excepción documentada al flujo «Drive si >100 MB» (ver ``transfer/README.md``):
los pesos son públicos en HuggingFace y su compuerta real es el SHA-256 fijado
en el manifiesto, no el transporte. Este script descarga exactamente los
archivos listados en el manifiesto (repositorio y revisión fijados por commit)
y delega la verificación en ``scripts/verify-model-manifest.py``.

Uso en la instancia (sin token; los repositorios son públicos):

    python scripts/download-wuerstchen-weights.py \
        --manifest models/wuerstchen-v2/model-manifest.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise SystemExit("ERROR: manifiesto de modelo inválido")

    from huggingface_hub import hf_hub_download

    destination_root = manifest.parent
    planned = sum(len(component["files"]) for component in payload["components"])
    downloaded = 0
    for component in payload["components"]:
        repository = component["repository"]
        revision = component["revision"]
        prefix = component["path"] + "/"
        for record in component["files"]:
            relative = record["path"]
            if not relative.startswith(prefix):
                raise SystemExit(f"ERROR: ruta fuera del componente: {relative}")
            filename = relative.removeprefix(prefix)
            path = hf_hub_download(
                repo_id=repository,
                repo_type="model",
                revision=revision,
                filename=filename,
                local_dir=destination_root / component["path"],
            )
            downloaded += 1
            print(f"download_ok {downloaded}/{planned} {relative} -> {path}", flush=True)

    verifier = Path(__file__).with_name("verify-model-manifest.py")
    subprocess.run([sys.executable, str(verifier), str(manifest)], check=True)


if __name__ == "__main__":
    main()
