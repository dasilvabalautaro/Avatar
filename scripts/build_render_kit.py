#!/usr/bin/env python3
"""Empaqueta el kit de integración del generador de avatares (ADR 0012).

Reúne la implementación en los dos lenguajes, el vocabulario, los casos de
referencia, la galería y la documentación, y produce un `.zip` autocontenido
que otro proyecto puede aplicar sin acceso a este repositorio.

El kit se **reconstruye**, no se versiona: así nunca queda desincronizado del
código. El hash del paquete se imprime al final y se registra en el HANDOFF.

Uso:
  python scripts/build_render_kit.py --output transfer/avatarface-render-kit.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Los módulos de Python del kit son planos: se reescriben los imports del
# paquete para que funcionen sin instalar `avatar-face`.
IMPORT_REWRITES = {
    "from avatar_face.domain.models import": "from prompt_models import",
    "from avatar_face.domain.attributes import": "from attributes import",
    "from avatar_face.infrastructure.rendering.geometry import": "from geometry import",
    "from avatar_face.infrastructure.rendering.palette import": "from palette import",
    "from avatar_face.infrastructure.rendering.avatar_renderer import": (
        "from avatar_renderer import"
    ),
}
SCRIPT_PREAMBLE = (
    "from __future__ import annotations\n\n"
    "import sys\n"
    "from pathlib import Path as _KitPath\n\n"
    "# El kit es plano: los módulos del dibujo viven en ../python.\n"
    'sys.path.insert(0, str(_KitPath(__file__).resolve().parent.parent / "python"))\n'
)


def rewrite(text: str, *, as_script: bool) -> str:
    for old, new in IMPORT_REWRITES.items():
        text = text.replace(old, new)
    if as_script:
        text = text.replace("from __future__ import annotations\n", SCRIPT_PREAMBLE, 1)
    return text


def build_vocabulary(destination: Path) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from avatar_face.domain.attributes import (  # noqa: PLC0415
        ATTRIBUTE_ORDER,
        ATTRIBUTE_VOCABULARIES,
        DEFAULT_ATTRIBUTES,
        LEGACY_ATTRIBUTES,
        vocabulary_size,
    )
    from avatar_face.domain.models import MINOR_AGE_TERMS  # noqa: PLC0415

    destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "combinaciones": vocabulary_size(),
                "orden_canonico": list(ATTRIBUTE_ORDER),
                "atributos_heredados": list(LEGACY_ATTRIBUTES),
                "por_defecto": DEFAULT_ATTRIBUTES,
                "vocabulario": {k: list(v) for k, v in ATTRIBUTE_VOCABULARIES.items()},
                "terminos_menores_rf09": sorted(MINOR_AGE_TERMS),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return vocabulary_size()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output", type=Path, default=ROOT / "transfer" / "avatarface-render-kit.zip"
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()

    work = args.work_dir or (ROOT / "artifacts" / "render-kit-build")
    kit = work / "avatarface-render-kit"
    if work.exists():
        shutil.rmtree(work)
    for name in ("kotlin", "python", "docs", "assets", "referencia", "scripts"):
        (kit / name).mkdir(parents=True)

    # Implementación nativa y de referencia.
    android = ROOT / "android/app/src/main/java/com/avatarface/app"
    for path in sorted((android / "render").glob("*.kt")):
        shutil.copy2(path, kit / "kotlin" / path.name)
    shutil.copy2(android / "AvatarActivity.kt", kit / "kotlin" / "AvatarActivity.kt")

    rendering = ROOT / "src/avatar_face/infrastructure/rendering"
    for name in ("geometry.py", "palette.py", "avatar_renderer.py"):
        (kit / "python" / name).write_text(
            rewrite((rendering / name).read_text(encoding="utf-8"), as_script=False),
            encoding="utf-8",
        )
    (kit / "python" / "attributes.py").write_text(
        rewrite(
            (ROOT / "src/avatar_face/domain/attributes.py").read_text(encoding="utf-8"),
            as_script=False,
        ),
        encoding="utf-8",
    )
    (kit / "python" / "prompt_models.py").write_text(
        (ROOT / "src/avatar_face/domain/models.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    for name in ("render_gallery.py", "dump_parser_cases.py"):
        (kit / "scripts" / name).write_text(
            rewrite((ROOT / "scripts" / name).read_text(encoding="utf-8"), as_script=True),
            encoding="utf-8",
        )
    shutil.copy2(
        ROOT / "scripts" / "compare_android_render.py",
        kit / "scripts" / "compare_android_render.py",
    )

    # Assets: vocabulario y casos de referencia, generados en el momento.
    combinations = build_vocabulary(kit / "assets" / "vocabulario.json")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "dump_parser_cases.py"),
         "--output", str(kit / "assets" / "parser-cases.json")],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "render_gallery.py"),
         "--output-dir", str(kit / "referencia"),
         "--dump-specs", str(kit / "assets" / "gallery-specs.json")],
        check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
    )
    for name in ("dispositivo-texto.png", "dispositivo-barba.png",
                 "dispositivo-rechazo-menor.png"):
        source = ROOT / "artifacts" / "render-android" / name
        if source.is_file():
            shutil.copy2(source, kit / "referencia" / name)

    # Documentación: la del kit más las licencias del proyecto.
    for name in ("LEEME.md", "por-que-sin-modelo.md", "vocabulario.md",
                 "reglas-de-estilo.md", "verificacion.md", "estructura.md"):
        source = ROOT / "docs" / "render-kit" / name
        target = kit / name if name == "LEEME.md" else kit / "docs" / name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("LICENSE", "NOTICE"):
        shutil.copy2(ROOT / name, kit / name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(kit.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(work))
    shutil.rmtree(work)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "sha256": digest,
                "combinaciones_vocabulario": combinations,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
