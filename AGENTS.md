# AGENTS.md — AvatarFace

Este archivo orienta a agentes de código que trabajen en este repositorio.
El idioma del proyecto (código, documentación y mensajes) es el español.

## Visión general del proyecto

AvatarFace genera rostros de avatar a partir de texto y está diseñado para
inferencia completamente offline en Android. El alcance actual es **sólo
Android** (ADR 0002). El proyecto está en la Fase 2: construcción de un dataset
legal, especializado y auditable, más un piloto LoRA sobre Würstchen v2 en GPU
remota (Vast.ai).

Puntos clave:

- El runtime provisional de inferencia es **ONNX Runtime CPU** con cuantización
  selectiva (ADR 0005 y 0006). NNAPI está descartado para el grafo actual.
- Los modelos de viabilidad tienen **pesos aleatorios**; validan contratos,
  exportación, cuantización y rendimiento, no calidad visual.
- El modelo base aprobado para entrenamiento real es **Würstchen v2 Stage C**
  (`warp-ai/wuerstchen-prior-model-base`), con pesos fijados por SHA-256 bajo
  `models/wuerstchen-v2/` (ignorado por Git).
- El plan completo está en `docs/plan-proyecto-avatarface.md` y el estado
  exacto para retomar el trabajo está en `docs/HANDOFF.md` — leer ese archivo
  primero al iniciar cualquier sesión de trabajo sustantiva.

## Estructura del repositorio

```text
src/avatar_face/
├── domain/           # Reglas de negocio puras; sin frameworks
│   ├── ports.py      # Contratos (Protocol) de todos los adaptadores
│   ├── models.py, feasibility.py, dataset.py, licensing.py,
│   │   benchmarking.py, exporting.py, quantization.py
├── application/      # Casos de uso; dependen sólo de puertos del dominio
├── infrastructure/   # Adaptadores: ADB, ONNX/PyTorch, Pillow, JSON
│   ├── android/      # adb_probe.py, adb_benchmark.py
│   ├── dataset/      # generador procedimental, auditor, loader, freezer
│   ├── feasibility/  # exportador y cuantizador ONNX, modelo torch sintético
│   └── training/     # microtrainer local reanudable
├── presentation/     # cli.py: entrada de operador (argparse)
android/              # App Kotlin instrumental (sin UI de producto)
configs/              # project.toml, candidatos de modelos, fuentes de dataset,
                      # fixtures de regresión congelados
scripts/              # Shell/Python auxiliares: transferencias Vast.ai,
                      # preflight, piloto LoRA Würstchen, manifiestos de pesos
tests/                # Pruebas pytest (una por módulo, prefijo test_)
docs/                 # Plan, ADRs, fases, datasheet, políticas, experimentos
artifacts/            # Salidas locales; ignoradas salvo .gitkeep
data/, models/        # Datasets y pesos locales; ignorados por Git
transfer/             # Paquetes .tar + SHA256SUMS para Drive/Vast; ignorados
```

## Arquitectura

Clean Architecture con dirección de dependencias fija
(`docs/architecture.md`):

```text
presentation ──→ application ──→ domain
                       ↑
infrastructure ────────┘
```

Reglas obligatorias:

1. El dominio no importa frameworks; los puertos son `typing.Protocol`.
2. La aplicación depende de puertos del dominio; la infraestructura los
   implementa.
3. Los casos de uso no imprimen ni leen argumentos CLI; eso es exclusivo de
   `presentation/cli.py`.
4. Los adaptadores convierten errores de proveedor en resultados explícitos
   (dataclasses tipadas).
5. Las rutas externas entran por configuración, nunca hardcodeadas en dominio.
6. Los formatos de artefactos incluyen `schema_version` y hashes SHA-256.
7. Un experimento no sobrescribe silenciosamente otro (bandera `--overwrite`).
8. Los imports de dependencias pesadas (torch, onnx, onnxruntime, Pillow) se
   hacen **de forma perezosa dentro de las funciones** de la CLI, para que la
   instalación base no los requiera.

## Stack tecnológico

- **Python** ≥ 3.11 (el entorno verificado usa 3.12.14 en `.venv`); paquete
  `avatar-face` con build backend hatchling, src-layout.
- **CLI**: `avatar-face` (entry point `avatar_face.presentation.cli:main`).
- **ML**: PyTorch 2.2.2, ONNX, ONNX Runtime (extras opcionales `feasibility`,
  `dataset`, `training`).
- **Android**: Kotlin, Gradle wrapper 9.4.1, Android Gradle Plugin 9.2.1,
  ONNX Runtime Android 1.23.2, compileSdk/targetSdk 35, minSdk 26, ABI única
  `arm64-v8a`, JDK Temurin 21.
- Dependencias resueltas fijadas en `requirements-dev.lock`,
  `requirements-feasibility.lock` y `requirements-dataset.lock`.

## Comandos de build, prueba y verificación

Entorno (el `pip` del host fuerza `--user`; usar siempre las banderas):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --no-user --no-cache-dir -e '.[dev]'
# Stacks opcionales: '.[feasibility]', '.[dataset]', '.[training]'
```

Validación estándar antes de dar por terminado cualquier cambio:

```bash
.venv/bin/pytest          # 40 pruebas
.venv/bin/ruff check .    # sin hallazgos
.venv/bin/mypy src        # estricto, sin errores
git diff --check
```

Build y benchmark Android (requieren dispositivo físico por USB):

```bash
cd android
JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home \
  ./gradlew --no-daemon :app:assembleDebug

# El serial es OBLIGATORIO: nunca ejecutar ADB sin -s
.venv/bin/avatar-face benchmark-android --serial 14254155BM000874 --runs 7
```

Comandos CLI disponibles (todos emiten resultados explícitos, la mayoría JSON):
`status`, `validate-prompt`, `audit-candidates`, `describe-feasibility`,
`export-feasibility`, `export-feasibility-components`, `quantize-feasibility`,
`benchmark-android`, `generate-smoke-dataset`, `generate-training-dataset`,
`audit-dataset`, `freeze-dataset`, `verify-frozen-dataset`, `train-smoke`,
`preflight-vast`.

## Estilo de código y convenciones

- Ruff: `line-length = 100`, `target-version = py311`, reglas `E, F, I, UP, B`.
- mypy en modo **estricto** sobre `src/`; todo el código nuevo debe tiparse.
- Estilo de tipos: `from __future__ import annotations`, `X | None`, genéricos
  integrados (`list[str]`, `tuple[...]`).
- Resultados y configuraciones como **dataclasses inmutables**; serialización
  con `asdict` + `json.dumps(..., ensure_ascii=False)`.
- Los casos de uso devuelven resultados tipados, no códigos de salida; los
  códigos de salida (0, 2, 3) se deciden en `presentation/cli.py`.
- Determinismo: seeds explícitas (por defecto 42), orden determinista,
  hashes SHA-256 de artefactos y manifiestos.
- Documentación, comentarios, docstrings y mensajes de usuario **en español**.
- Cada ADR en `docs/adr/` registra una decisión arquitectónica; crear uno nuevo
  cuando se cambie una decisión existente (no editar la historia).

## Estrategia de pruebas

- pytest con `pythonpath = ["src"]` y `testpaths = ["tests"]`, `addopts = -q`.
- Las pruebas no requieren torch/onnx/Pillow: las capas pesadas se prueban con
  dobles o se omiten; la lógica de dominio y aplicación se cubre siempre.
- Los inputs de regresión (captions, prompts, seeds) están **congelados** en
  `configs/regression-fixtures.json` junto con `tests/test_regression_fixtures.py`.
  Todo cambio intencional debe actualizar ambos.
- Métricas de rendimiento válidas **sólo desde dispositivo físico** (TECNO
  KM5s, Android 15); los resultados de emulador no son métricas finales.

## Seguridad, licencias y datos

- **Compuerta de licencias** (`docs/license-policy.md`): sólo se admiten
  automáticamente Apache-2.0, MIT, BSD-2/3-Clause y CC0-1.0. Se rechazan
  RAIL/OpenRAIL, research-only, revisiones mutables sin hash fijado y
  procedencia desconocida. Cada componente registra URL, revisión, licencia y
  SHA-256.
- Código y documentación propios: Apache-2.0 (`LICENSE`, `NOTICE`). Imágenes,
  captions y manifiestos del dataset procedimental propio: CC0-1.0
  (`docs/dataset/CC0-DEDICATION.md`).
- **Prohibido usar rostros reales** en datasets; el generador es procedimental
  (Pillow) y la auditoría valida campos legales, hashes y duplicados.
- **Prohibido generar, entrenar o validar avatares de menores de edad**: el
  producto es sólo para adultos (RF-09 en `docs/product-requirements.md`,
  riesgo R-16). El filtro de prompts, las plantillas de captions y la
  evaluación visual deben aplicar esta restricción.
- **Nunca** commitear credenciales, tokens, enlaces gated, IDs privados de
  Drive/Vast ni comandos SSH con secretos.
- Qué no se versiona (ver `.gitignore`): `.venv/`, `data/`, `models/`,
  `downloads/`, `artifacts/` (salvo `.gitkeep`), `transfer/*.tar` y sus
  checksums, builds Android, APK/AAB, `.env`.
- Todo ADB debe usar el serial explícito del dispositivo (`-s`).
- Flujo Vast.ai obligatorio: máquina local → `.tar` + SHA256SUMS → (Drive si
  >100 MB) → restauración verificada por hash; los scripts en `scripts/`
  (`package-for-drive.sh`, `restore-from-drive.sh`, `preflight-vast.sh`)
  rechazan rutas inseguras y no suben ni transfieren automáticamente.
- `git push` lo realiza el usuario; los agentes no empujan cambios.
