# Handoff de AvatarFace

Actualizado: 2026-08-13, zona horaria `America/La_Paz`.

Este documento es el punto de entrada obligatorio para retomar el proyecto en
otra sesión. El estado descrito corresponde al commit que contiene este archivo;
su SHA se obtiene con `git log -1 --oneline`.

## 1. Estado ejecutivo

AvatarFace busca generar rostros de avatar desde texto, completamente offline en
Android. Sólo Android está dentro del alcance actual. El dispositivo físico de
referencia está conectado por USB.

Completado:

- fundamentos, Clean Architecture, CLI y compuerta de licencias de modelos;
- spike temprano PyTorch → ONNX → APK → ONNX Runtime Android;
- cuantización INT8 estática QDQ y preprocesamiento ONNX;
- benchmark ADB reproducible con serial explícito;
- perfilado por componentes y por operador;
- pipeline móvil selectivo persistente validado en dispositivo;
- smoke dataset procedimental, auditoría, loader reproducible y
  microentrenamiento local reanudable.

En curso:

- Fase 2 — dataset legal, especializado y auditable; release procedimental v2
  congelada y smoke remoto completado.

Próxima tarea exacta:

> Auditar los candidatos y seleccionar un modelo base permisivo; después,
> redactar el plan de LoRA/fine-tuning en GPU sobre la release v2 congelada.

Actualización: Würstchen v2 Stage C base fue aprobado, descargado localmente y
empaquetado con hashes. El próximo paso es subir el paquete a Drive y, en una
nueva instancia Vast, restaurar/verificar pesos antes de un piloto LoRA de 20
pasos.

## 2. Repositorio y entorno

- Workspace: `/Users/davidsilva/VisualStudioCodeProjects/Avatar`.
- Rama: `main`.
- Remote: `origin`, repositorio GitHub `dasilvabalautaro/Avatar`.
- El usuario realiza el push; no hacer push automáticamente.
- Python verificado: 3.12.14 en `.venv`.
- Host: macOS Intel.
- JDK Android:
  `/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home`.
- Gradle wrapper: 9.4.1.
- Android Gradle Plugin: 9.2.1.
- ONNX Runtime Android: 1.23.2.
- ABI del APK: sólo `arm64-v8a`.
- Dispositivo: TECNO KM5s, Android 15/API 35, MT6769, 3.66 GiB RAM.
- Serial ADB obligatorio: `14254155BM000874`.

Pip tiene una configuración externa que intenta usar `--user`. Al instalar en
`.venv`, usar siempre `--no-user --no-cache-dir`. Las descargas reconstruibles no
deben conservarse.

## 3. Validación vigente

En el momento de este handoff:

- pytest: 38 pruebas superadas;
- Ruff: sin hallazgos;
- mypy estricto: sin errores en 33 archivos fuente;
- Gradle `:app:assembleDebug`: exitoso;
- smoke dataset: 64 muestras, 64 hashes únicos, cero hallazgos;
- no quedó ningún proceso `com.avatarface.app` activo en el teléfono.

Comandos de control:

```bash
cd /Users/davidsilva/VisualStudioCodeProjects/Avatar
.venv/bin/ruff check .
.venv/bin/mypy src
.venv/bin/pytest
git diff --check

cd android
JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home \
  ./gradlew --no-daemon :app:assembleDebug
```

## 4. Baseline Android válido

Runtime provisional: ONNX Runtime CPU. NNAPI queda descartado para el grafo
actual por fragmentación severa y timeouts.

Pipeline selectivo persistente:

| Componente | Precisión | Tamaño |
|---|---|---:|
| Encoder `bridge-encoder` | FP32, salida cacheada por prompt | 2,262,846 B |
| Denoiser `bridge-denoiser-int8-preprocessed` | INT8 QDQ | 8,224,426 B |
| Decoder `bridge-fast-decoder` | FP32 | 32,747 B |
| Total | Mixta | 10,520,019 B |

Serie formal de 30 corridas en TECNO KM5s:

- mínimo: 58.6 ms;
- P50: 67.5 ms;
- media: 66.9 ms;
- P90: 73.1 ms;
- P95: 78.1 ms;
- máximo: 78.9 ms;
- encoder ejecutado una vez: 1.45 ms;
- creación de tres sesiones: 532.2 ms;
- PSS máximo muestreado: 162,859 KiB;
- temperatura HAL posterior: 27.6 °C, Thermal Status 0.

La latencia mide exclusivamente `session.run` para denoiser→decoder. Checksum,
serialización y lectura del tensor están fuera del cronómetro.

Advertencia metodológica importante: los primeros benchmarks incluían el
checksum RGB dentro del cronómetro. Fueron repetidos y los informes corregidos.
No recuperar cifras antiguas desde el historial sin leer las notas de validez.

Evidencia principal:

- `docs/experiments/android-bridge-selective-pipeline-2026-08-13.md`;
- `docs/experiments/android-bridge-components-2026-08-13.md`;
- `docs/mobile-feasibility-status.md`;
- ADR 0005 y ADR 0006.

Repetición:

```bash
.venv/bin/avatar-face benchmark-android \
  --serial 14254155BM000874 \
  --model-asset avatarface-feasibility-bridge-selective.onnx \
  --backend cpu \
  --runs 30
```

El nombre `bridge-selective.onnx` es un identificador virtual del benchmark; la
app conecta los tres assets indicados en la tabla.

## 5. Modelos de viabilidad

Los pesos son aleatorios y no generan avatares útiles. Sirven exclusivamente para
validar contratos, exportación, cuantización y rendimiento.

Perfiles disponibles:

- `micro`: 304,415 parámetros;
- `bridge`: 8,238,551 parámetros;
- `bridge-slim`: mismo encoder/denoiser y decoder reducido;
- `bridge-fast`: decoder de tres escalas; pipeline FP32 integrado 84.8 ms;
- `target`: aproximadamente 136 M parámetros; no desplegar aún;
- `stress`: reservado para límites posteriores.

Decisiones:

- encoder: FP32 y cacheado;
- denoiser: INT8 sólo tras recalibrar con tensores representativos;
- decoder experimental: `bridge-fast` FP32;
- mantener sesiones vivas; el denoiser INT8 domina el arranque;
- no escalar a `target` antes del microentrenamiento y la evaluación visual.

## 6. Dataset actual

Ruta local ignorada por Git:
`data/smoke-procedural/`.

Contenido generado:

- 64 PNG RGB de 256 × 256;
- train 50, validation 7, test 7;
- manifiesto: `data/smoke-procedural/manifest.json`;
- SHA-256 del manifiesto:
  `146026fb9c4e99ed92ac7ba359b35ff7c9aee69c5f9fd80f29874cac672b7ae2`;
- tamaño aproximado: 316 KiB;
- 64 hashes de imagen únicos;
- sin personas reales, modelos generativos ni assets externos.

El loader consume exclusivamente el manifiesto y el preflight valida los hashes
de todas las muestras. La corrida local ejecutó 5 pasos y fue reanudada hasta el
paso 7 con el mismo hash de manifiesto. El checkpoint y la reconstrucción de
validation viven en `artifacts/training/` y no se versionan.

Reproducción y auditoría:

```bash
.venv/bin/avatar-face generate-smoke-dataset \
  --output-dir data/smoke-procedural \
  --samples 64 \
  --seed 42 \
  --overwrite

.venv/bin/avatar-face audit-dataset \
  --manifest data/smoke-procedural/manifest.json
```

La auditoría valida rutas, campos legales, bandera sintética, archivos, hashes,
IDs, duplicados exactos y similitud perceptual RGB.

Imágenes y metadatos procedimentales: dedicados a CC0-1.0 con confirmación del
titular en `docs/dataset/CC0-DEDICATION.md`. El código y documentación propios
del repositorio se publican bajo Apache-2.0 (`LICENSE` y `NOTICE`). Estas
decisiones no cubren dependencias, modelos, pesos ni otros activos de terceros.

Documentos:

- `docs/dataset/datasheet-smoke-procedural.md`;
- `docs/dataset/license-matrix.md`;
- `docs/phase-2-dataset-status.md`.

Limitaciones: estilo geométrico simple, captions en inglés por plantilla,
distribución no perfectamente balanceada y sin detección perceptual de duplicados.

## 7. Artefactos ignorados y regeneración

No añadir a Git:

- `.venv/`, cachés Python y Gradle;
- `data/`, `models/`, `downloads/`;
- `artifacts/` salvo `.gitkeep`;
- builds Android, APK y AAB;
- paquetes `.tar` de transferencia.

Artefactos locales importantes pero ignorados:

- ONNX y manifiestos: `artifacts/feasibility/`;
- JSON y perfiles ORT Android: `artifacts/android/`;
- smoke dataset: `data/smoke-procedural/`;
- APK: `android/app/build/outputs/apk/debug/app-debug.apk`.

Regenerar el pipeline de viabilidad:

```bash
.venv/bin/avatar-face export-feasibility --profile bridge-fast --overwrite
.venv/bin/avatar-face export-feasibility-components --profile bridge --overwrite
.venv/bin/avatar-face export-feasibility-components --profile bridge-fast --overwrite
.venv/bin/avatar-face quantize-feasibility \
  --source artifacts/feasibility/avatarface-feasibility-bridge-denoiser.onnx \
  --overwrite
```

## 8. Arquitectura implementada

- `domain`: perfiles, dataset, contratos, resultados y políticas.
- `application`: casos de uso de auditoría, exportación, cuantización, dataset y
  benchmark.
- `infrastructure`: ADB, ONNX/PyTorch, cuantizador, generador Pillow y auditor
  JSON.
- `presentation`: CLI `avatar-face`.
- `android`: app Kotlin instrumental sin UI de producto.

Comandos CLI relevantes:

```text
status
validate-prompt
audit-candidates
describe-feasibility
export-feasibility
export-feasibility-components
quantize-feasibility
benchmark-android
generate-smoke-dataset
audit-dataset
```

Dependencias fijadas:

- `requirements-dev.lock`;
- `requirements-feasibility.lock`;
- `requirements-dataset.lock`.

## 9. Próximas tareas, en orden estricto

### P0. Loader reproducible

1. Crear entidad/configuración tipada para dataset y entrenamiento.
2. Implementar loader desde `manifest.json`; nunca escanear imágenes sin
   manifiesto.
3. Verificar SHA-256 opcionalmente al abrir cada muestra y siempre en preflight.
4. Aplicar splits declarados, seed fija y orden determinista.
5. Normalizar PNG a tensores `[-1, 1]` sin augmentations en la primera prueba.
6. Añadir pruebas de batch, shape, split, hash alterado y determinismo.

Criterio de salida alcanzado: un batch train reproducible de forma
`[B, 3, 256, 256]`, normalizado a `[-1, 1]`, y ninguna muestra fuera del
manifiesto.

### P0. Microentrenamiento local reanudable

1. Implementar primero un autoencoder/decoder pequeño; no entrenar todavía el
   text-to-image completo.
2. Configurar una corrida de 5–20 steps sobre CPU/local para probar el plumbing.
3. Guardar checkpoint con modelo, optimizer, step, seed, config y hash del
   manifiesto.
4. Reanudar y demostrar que continúa desde el step correcto.
5. Guardar reconstrucciones de validation bajo `artifacts/`, nunca en Git.
6. Registrar pérdida y tiempo; calidad visual no es aún compuerta.

Criterio de salida alcanzado: checkpoint reanudable y ejecución determinista sin
error. Comando: `avatar-face train-smoke --steps 5`, seguido de
`avatar-face train-smoke --steps 7 --resume`.

### P0. Mejorar dataset antes de entrenamiento serio

1. Muestreo estratificado, geometría facial y similitud perceptual: completados.
2. Regresión congelada y licencias de publicación formalizadas.

### P1. Vast.ai, sólo después del smoke local

1. Inspeccionar y reutilizar patrones del proyecto
   `/Users/davidsilva/Opencode/Anthropic/MythosLight`.
2. Implementar `preflight-vast`; no iniciar entrenamiento costoso directamente.
3. No descargar desde Vast.ai. Flujo obligatorio:
   máquina local → archivo `.tar` → Google Drive → descarga desde Drive en Vast.
4. Calcular SHA-256 antes y después de transferir.
5. Eliminar descargas y `.tar` reconstruibles después de verificarlos y usarlos.
6. No registrar credenciales, IDs privados ni comandos SSH con secretos.

## 10. Riesgos y prohibiciones vigentes

- No usar modelos, encoders, datasets o pesos con restricciones de uso.
- No asumir que la licencia del repositorio cubre todos los componentes.
- No usar rostros reales en el smoke dataset.
- No aceptar métricas de emulador como métricas finales.
- No ejecutar ADB sin `-s 14254155BM000874`.
- No volver a incluir checksum/posproceso en la latencia de inferencia.
- No usar NNAPI en el baseline actual.
- No desplegar `target` todavía.
- No hacer push; el usuario lo realiza.
- No conservar descargas reconstruibles.

## 11. Primeros comandos de la próxima sesión

```bash
cd /Users/davidsilva/VisualStudioCodeProjects/Avatar
sed -n '1,260p' docs/HANDOFF.md
git status --short
git log -1 --oneline
.venv/bin/pytest
.venv/bin/avatar-face audit-dataset \
  --manifest data/smoke-procedural/manifest.json
```

Después, comenzar directamente por `P0. Loader reproducible` de este documento.
