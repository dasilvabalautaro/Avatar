#!/usr/bin/env python3
"""Escribe evidencia de procedencia y hashes para los pesos locales aprobados."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = ROOT / "models" / "wuerstchen-v2"
OUTPUT = MODEL_DIRECTORY / "model-manifest.json"
COMPONENTS = (
    {
        "id": "prior-base",
        "path": "prior-base",
        "repository": "warp-ai/wuerstchen-prior-model-base",
        "revision": "3f9205c8c2e7cf103192954fe6f096e66f9d4efc",
        "license_id": "MIT",
    },
    {
        "id": "decoder",
        "path": "decoder",
        "repository": "warp-ai/wuerstchen",
        "revision": "c3da41406ddd4d9c48c49aa93981a82354351b83",
        "license_id": "MIT",
    },
    {
        "id": "text-encoder",
        "path": "text-encoder",
        "repository": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "revision": "743c27bd53dfe508a0ade0f50698f99b39d03bec",
        "license_id": "MIT",
    },
    {
        "id": "stage-b-encoder",
        "path": "stage-b-encoder",
        "repository": "dome272/wuerstchen",
        "revision": "6eb6cf5494fa67472afdbcfd78a31dcb091b0c05",
        "license_id": "MIT",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    records = []
    for component in COMPONENTS:
        directory = MODEL_DIRECTORY / component["path"]
        if not directory.is_dir():
            raise SystemExit(f"Falta componente: {directory}")
        files = []
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            if path.suffix == ".bin" or ".cache" in path.parts:
                raise SystemExit(f"Archivo no permitido en release: {path}")
            files.append(
                {
                    "path": str(path.relative_to(MODEL_DIRECTORY)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        records.append({**component, "files": files})
    payload = {"schema_version": 1, "components": records}
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(sha256(OUTPUT))


if __name__ == "__main__":
    main()
