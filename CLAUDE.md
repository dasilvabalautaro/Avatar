# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

El idioma del proyecto (código, documentación, commits y mensajes) es el **español**.

Antes de cualquier trabajo sustantivo, leer `docs/HANDOFF.md` (estado exacto del
proyecto) y `AGENTS.md` (guía completa para agentes; este archivo la resume).
El plan general está en `docs/plan-proyecto-avatarface.md`.

## Qué es este proyecto

AvatarFace genera rostros de avatar a partir de texto, con inferencia
completamente offline en Android (alcance **sólo Android**, ADR 0002). Runtime
provisional: ONNX Runtime CPU con cuantización selectiva. Modelo base aprobado
para entrenamiento real: Würstchen v2 Stage C, pesos fijados por SHA-256 bajo
`models/wuerstchen-v2/` (ignorado por Git). Fase actual: dataset legal y
auditable + experimentos LoRA/destilación en GPU remota (Vast.ai).

## Comandos

Entorno (el `pip` del host fuerza `--user`; usar siempre las banderas):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --no-user --no-cache-dir -e '.[dev]'
# Stacks opcionales: '.[feasibility]', '.[dataset]', '.[training]'
```

Validación estándar antes de dar por terminado cualquier cambio:

```bash
.venv/bin/pytest          # todas las pruebas, sin fallos
.venv/bin/ruff check .    # sin hallazgos
.venv/bin/mypy src        # estricto, sin errores
git diff --check
```

Una sola prueba: `.venv/bin/pytest tests/test_<modulo>.py` (o `-k <patrón>`).

CLI del proyecto: `avatar-face <comando>` (entry point
`avatar_face.presentation.cli:main`); ver la lista completa con
`avatar-face --help`. La mayoría emite JSON con `--json`.

Build y benchmark Android (requieren dispositivo físico por USB):

```bash
cd android
JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home \
  ./gradlew --no-daemon :app:assembleDebug

# El serial es OBLIGATORIO: nunca ejecutar ADB sin -s
.venv/bin/avatar-face benchmark-android --serial 14254155BM000874 --runs 7
```

## Arquitectura

Clean Architecture con dirección de dependencias fija (`docs/architecture.md`):

```text
presentation ──→ application ──→ domain
                       ↑
infrastructure ────────┘
```

- `src/avatar_face/domain/`: reglas de negocio puras, sin frameworks; los
  contratos de todos los adaptadores son `typing.Protocol` en `ports.py`.
- `src/avatar_face/application/`: casos de uso; dependen sólo de puertos del
  dominio, devuelven dataclasses tipadas, nunca imprimen ni leen argv.
- `src/avatar_face/infrastructure/`: adaptadores (ADB, ONNX/PyTorch, Pillow,
  JSON) que implementan los puertos y convierten errores de proveedor en
  resultados explícitos.
- `src/avatar_face/presentation/cli.py`: único lugar con argparse, prints y
  códigos de salida (0, 2, 3).
- Los imports pesados (torch, onnx, onnxruntime, Pillow) se hacen **de forma
  perezosa dentro de las funciones**, para que la instalación base no los
  requiera; las pruebas tampoco los necesitan (dobles o skip).

Convenciones clave:

- mypy estricto; `from __future__ import annotations`, `X | None`, genéricos
  integrados; resultados como dataclasses inmutables serializadas con
  `asdict` + `json.dumps(..., ensure_ascii=False)`.
- Determinismo: seeds explícitas (por defecto 42), orden determinista, hashes
  SHA-256 y `schema_version` en todo artefacto/manifiesto; un experimento no
  sobrescribe otro sin `--overwrite`.
- Los inputs de regresión (captions, prompts, seeds) están **congelados** en
  `configs/regression-fixtures.json` + `tests/test_regression_fixtures.py`;
  todo cambio intencional actualiza ambos.
- Decisiones arquitectónicas en `docs/adr/`; para cambiar una decisión se crea
  un ADR nuevo, no se edita la historia.

## Reglas no negociables

- **Compuerta de licencias** (`docs/license-policy.md`): sólo Apache-2.0, MIT,
  BSD-2/3-Clause y CC0-1.0; se rechazan RAIL/OpenRAIL, research-only y
  revisiones sin hash fijado.
- **Prohibido usar rostros reales** en datasets (generador procedimental) y
  **prohibido generar, entrenar o validar avatares de menores** (RF-09): el
  rechazo está implementado en `AvatarPrompt` y las captions marcan
  «of an adult».
- Nunca commitear credenciales, `keys-git.md`, enlaces gated, IDs de
  Drive/Vast ni secretos; `data/`, `models/`, `artifacts/`, `transfer/*.tar`
  y builds Android no se versionan.
- Métricas de rendimiento válidas sólo desde el dispositivo físico (TECNO
  KM5s); nunca desde emulador.
- Transferencias a Vast.ai por el flujo verificado por hash de
  `transfer/README.md` y `scripts/`; excepción: pesos públicos de HuggingFace
  fijados por SHA-256 pueden descargarse directo en la instancia con
  `scripts/download-wuerstchen-weights.py`.
