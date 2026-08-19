# Handoff de AvatarFace

Actualizado: 2026-08-19, zona horaria `America/La_Paz`.

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
  microentrenamiento local reanudable;
- release de entrenamiento `training-procedural-v2` v2.0.0 congelada y
  smoke remoto en Vast completado;
- Würstchen v2 Stage C aprobado, descargado y fijado por SHA-256;
- piloto LoRA de 20 pasos ejecutado en RTX 4090 (integración correcta; la
  validación de calidad era inválida por configuración y fue descartada);
- validador `scripts/validate_wuerstchen_lora.py` corregido a la receta
  oficial (30 timesteps, fp16, 1024, guía 8.0, prompt negativo);
- pesos Stage B restaurados en Vast con SHA-256 completo verificado
  (`c2519641...`) vía descarga autenticada de Drive con rclone/OAuth;
- compuerta `base-only` superada: rostro de avatar válido a 1024 px
  (`docs/experiments/wuerstchen-base-only-2026-08-15.md`);
- piloto LoRA v3 de 20 pasos validado de extremo a extremo con el validador
  corregido (`docs/experiments/wuerstchen-lora-pilot-v3-2026-08-15.md`);
- loader del encoder de texto corregido en
  `scripts/run_wuerstchen_lora_pilot.py` (prefijo `text_model.` y búfer
  `position_ids`);
- filtro de sólo adultos (RF-09) implementado: `AvatarPrompt` rechaza prompts
  de menores, las plantillas de captions marcan «of an adult» y los fixtures
  de regresión incluyen prompts de menores rechazados;
- diseño del experimento LoRA escala-1 fijado (`docs/lora-scale-1-design.md`);
- ADR 0007: brecha entre la receta oficial y el presupuesto móvil registrada;
  la integración móvil del modelo real exige destilación o reducción de pasos;
- experimento LoRA escala-1 completado y validado visualmente: 200 pasos con
  `lr=5e-5`, 16 muestras válidas (8 LoRA + 8 base-only), todas de adultos;
  pesos restaurados por descarga directa verificada SHA-256
  (`docs/experiments/wuerstchen-lora-scale-1-2026-08-15.md`);
- experimento LoRA escala-2 completado y validado visualmente: 500 pasos con
  `lr=5e-5` en una instancia nueva vía re-entrada automatizada
  (`scripts/bootstrap-vast.sh`), 16 muestras válidas, todas de adultos;
  divergencia clara vs. base-only y vs. escala-1, pero la fidelidad de
  atributos difíciles (ojos, pecas, accesorios) ya no mejora con más pasos —
  el límite es el dataset, no el entrenamiento
  (`docs/experiments/wuerstchen-lora-scale-2-2026-08-16.md`);
- release de dataset **v2.1.0** generada, auditada y congelada (2026-08-16):
  generador `avatarface-procedural-v3` con captions «of an adult» que detallan
  la forma de los ojos (`almond`/`round`/`narrow`) y amplían los accesorios
  (gafas cuadradas, gafas de sol); 1024 muestras (train 818, validation 103,
  test 103), lock SHA-256 y paquete de transferencia directa listos
  (`docs/phase-2-dataset-status.md`).
- experimento LoRA escala-3 completado y validado visualmente (2026-08-17):
  500 pasos con `lr=5e-5` sobre la release **v2.1.0** en una instancia nueva
  vía re-entrada automatizada (`scripts/bootstrap-vast.sh`), 16 muestras
  válidas, todas de adultos; el artefacto de texto del caso 2 desapareció,
  pero la fidelidad de atributos difíciles (ojos, pecas, pendientes) tampoco
  mejoró con el doble de muestras y captions más finos — el cuello de botella
  ya no es el dataset (`docs/experiments/wuerstchen-lora-scale-3-2026-08-17.md`).
- decisión del usuario registrada (2026-08-17): probar en una sola corrida la
  subida de resolución de entrenamiento y la ampliación del LoRA antes de
  aceptar el techo del piloto; diseño fijado en `docs/lora-scale-4-design.md`;
- experimento LoRA escala-4 completado y validado visualmente (2026-08-17):
  500 pasos, `lr=5e-5`, 512 px sobre la release v2.2.0, LoRA rango 32 con
  módulos atención+FFN (12,582,912 parámetros entrenables), 16 muestras
  válidas, todas de adultos; **ni la resolución ni la capacidad compraron
  fidelidad en color de ojos, pecas ni pendientes** (señal parcial: iris
  ámbar en el caso 5; artefactos nuevos por memorización de la franja de
  firma). El techo está en el modelo base para estos rasgos finos; conforme a
  la compuerta del diseño, no se pagan más corridas de este tipo y la
  siguiente fase es integración con destilación (ADR 0007)
  (`docs/experiments/wuerstchen-lora-scale-4-2026-08-17.md`);
- dibujo del generador procedimental escalado a `image_size` con salida a
  256 px verificada bit-idéntica (hashes v1.1.0 y v2.1.0 sin cambio) y CLI
  `generate-training-dataset` con `--image-size`;
- `scripts/run_wuerstchen_lora_pilot.py` parametrizado (`--resolution`,
  `--lora-rank`, `--lora-alpha`, `--lora-dropout`, `--lora-modules`; la config
  viaja en el checkpoint) y `scripts/validate_wuerstchen_lora.py` la respeta,
  con respaldo a los valores históricos de escala 1-3;
- release de dataset **v2.2.0** generada, auditada y congelada (2026-08-17):
  1024 muestras nativas de 512 × 512, lock `dataset-2.2.0.lock.json`,
  manifiesto SHA-256
  `53fa3b374dc1ed48e59a4510b2348fd43ef6c12027420ccc0fa69e4f4a2616f8`,
  paquete `transfer/avatarface-training-procedural-v2-2.tar` (7,171,584 bytes,
  SHA-256 `47744cf589e1160c50d22e852cf7886caebfa2c8fef50ea49f0b66a9c0026fd6`);
- `scripts/bootstrap-vast.sh` acepta el dataset a restaurar como argumentos
  opcionales (por defecto sigue restaurando v2.1.0).

- ADR 0008 aceptado (2026-08-17): la integración móvil del modelo real será
  **destilación progresiva del prior a 256 px** (30→8 pasos en dos etapas,
  maestro = LoRA escala-3), precedida de una línea base sin entrenamiento
  (scheduler reducido a 12 y 8 timesteps); presupuesto de validación fijado
  (`docs/adr/0008-destilacion-progresiva-prior-256.md`);
- línea base del ADR 0008 ejecutada (2026-08-17): reducir timesteps sin
  entrenar **no es viable** (12 pasos → 2/8 rostros válidos; 8 pasos → 0/8);
  la destilación progresiva es obligatoria
  (`docs/experiments/wuerstchen-baseline-steps-2026-08-17.md`);
- destilación etapa 1 (30→15 pasos, peso SNR) ejecutada el 2026-08-18:
  **compuerta no superada (0/8)**; el peso SNR anulaba el gradiente en saltos
  de alto ruido (`docs/experiments/wuerstchen-distill-stage-1-2026-08-18.md`);
- destilación etapa 1b (30→15 pasos, **peso uniforme**) ejecutada el
  2026-08-18: **compuerta no superada (1/8)**; la señal de gradiente quedó
  restaurada (sin pérdidas casi nulas) y la muestra 03 es un rostro válido,
  pero 2000 pasos con `lr=1e-5` no bastan para aprender los 14 saltos con la
  precisión que exigen los rasgos
  (`docs/experiments/wuerstchen-distill-stage-1b-2026-08-18.md`);
- destilación etapa 1c (30→15 pasos, **objetivo normalizado** mse/‖eps‖²,
  `lr=5e-5`, 6000 pasos) ejecutada el 2026-08-18: **compuerta no superada
  (0/8)**; las 8 muestras contienen rostros con rasgos completos (1b lograba
  1/8) pero siempre **repetidos en mosaico** — el estudiante sobre-actúa en
  cada salto. Agotadas las correcciones baratas del objetivo, la vía de
  destilación progresiva queda **descartada** salvo reformulación del método
  (`docs/experiments/wuerstchen-distill-stage-1c-2026-08-18.md`).
- decisión del usuario registrada en **ADR 0009** (2026-08-18): la integración
  móvil sigue por **destilación sobre trayectorias reales del maestro**
  (opción (a); (b) queda como cuantización posterior del artefacto y (c)
  descartada mientras quede vía offline). Diseño fijado en
  `docs/distill-trajectory-design.md` y `scripts/distill_wuerstchen_prior.py`
  extendido con `--mode trajectory` (genera las trayectorias guiadas del
  maestro una vez por caption, cachea los embeddings de texto y entrena sin
  maestro); `--mode marginal` se conserva para reproducibilidad histórica;
- aclaración de alcance del usuario (2026-08-18): el TECNO KM5s es sólo el
  dispositivo de referencia para pruebas; el producto debe funcionar en
  **Android API 31 a 36** (cubierto por minSdk 26 y ONNX Runtime 1.23.2).
- destilación etapa 1d (30→15 pasos, **trayectorias reales del maestro**;
  ADR 0009) ejecutada el 2026-08-18 en una instancia nueva vía
  `scripts/bootstrap-vast.sh`: **compuerta no superada (0/8)**; el defecto de
  repetición (rostros/rasgos apilados) persiste entrenando sobre la
  distribución real del maestro — la vía de destilación progresiva del prior
  de 994 M queda **cerrada con evidencia** (cuatro formulaciones: SNR,
  uniforme, normalizado, trayectorias)
  (`docs/experiments/wuerstchen-distill-stage-1d-2026-08-18.md`).

- decisión del usuario registrada en **ADR 0010** (2026-08-19): se mantiene el
  alcance offline y la vía es la **reducción arquitectónica** — estudiante
  compacto (U-Net 30–60 M, 256 px, condicionamiento por atributos sin encoder
  de texto de terceros) entrenado desde cero sobre salidas del maestro
  (LoRA escala-3, receta oficial); tope de fase 40 h de RTX 4090; diseño y
  presupuesto por etapas en `docs/student-distill-design.md`;
- etapa E1 del ADR 0010 completada en local (2026-08-19), sin GPU:
  parser de atributos con vocabulario cerrado v3 y filtro RF-09
  (`domain/attributes.py`), muestreador determinista de pares de destilación
  (`domain/distill.py` + comando `generate-distill-captions`), U-Net del
  estudiante de 52,231,267 parámetros por defecto
  (`infrastructure/training/student_unet.py`), fuente `AF-DISTILL-001`
  registrada en `configs/dataset-sources.json`, y scripts GPU reanudables
  `scripts/generate_distill_dataset.py` (maestro → manifiesto auditable) y
  `scripts/train_student.py` (formulaciones `direct` y `diffusion`, EMA,
  muestras de control); bucle de entrenamiento e inferencia verificado en
  smoke CPU local.

- etapa E2 del ADR 0010 completada (2026-08-19): release
  `avatarface-distill-teacher` v1.0.0 generada por el maestro (LoRA escala-3,
  receta oficial) en una RTX 4090 — 4096 muestras a 256 px, auditoría con
  4096 hashes únicos y cero hallazgos, lock congelado, manifiesto SHA-256
  `05f36cb10f99efbbc4e34bcf36fa3274a8dcb0c9c348f3ed3fd6682df9300a26`,
  paquete `transfer/avatarface-distill-teacher-v1.tar` (154,613,760 bytes,
  SHA-256 `ed0fc1d87e6e6f9f2c2919fa2266428ed3b5461e1d28b826d9d555196a5c0a1b`)
  bajado directo y verificado en local (~2 s por muestra; costo ~1–2 USD)
  (`docs/experiments/distill-teacher-dataset-2026-08-19.md`).

- etapa E3 con la **formulación A (directa) descartada** (2026-08-19):
  detenida en el paso 25,000 de 50,000. Aprendió toda la estructura de baja
  frecuencia (silueta, pelo, piel, fondo) pero **ningún rasgo facial**; el
  diagnóstico sobre muestras del split **train** mostró que no ajusta ni los
  datos de entrenamiento (no es brecha de generalización) — patrón propio de
  la regresión L1 con entrada de ruido puro. Evidencia archivada con hashes
  en `artifacts/student-direct-1/`
  (`docs/experiments/student-direct-2026-08-19.md`). Nota operativa: batch 32
  en fp32 excede la VRAM de la 4090; batch 16 usa 40.4 GiB.

En curso:

- Fase 3 — etapa E4 del ADR 0010, **formulación B (difusión) corriendo en la
  instancia Vast** (2026-08-19): `scripts/train_student.py --formulation
  diffusion --batch-size 16`, 50,000 pasos, lr 1e-4, misma release v1.0.0 y
  semilla 42; muestreo DDIM de 8 pasos, checkpoint reanudable y muestras de
  control cada 5,000 pasos en `artifacts/student-diffusion-1/`, log en
  `/workspace/student-diffusion-1.log` (~14 h). Es la **iteración única** de
  respaldo del punto 4 del ADR 0010. **La instancia debe seguir encendida
  hasta que termine.**

Requisito rector nuevo (2026-08-15): el producto genera **sólo rostros de
adultos**; está prohibido generar avatares de menores de edad. Ver RF-09 en
`docs/product-requirements.md`, riesgo R-16 y la sección 11.

Próxima tarea exacta:

> Esperar el fin de la **formulación B (difusión)**, en curso en la instancia
> Vast (~14 h). Vigilar las muestras de control: si a los 10,000–15,000 pasos
> siguen sin aparecer rasgos faciales, diagnosticar como en la formulación A
> (reconstrucción de captions del split train) antes de agotar la corrida. Al
> terminar: bajar `artifacts/student-diffusion-1/` (checkpoint y muestras),
> evaluar la **compuerta visual ≥ 6/8** con rostros de adultos y fidelidad de
> atributos comparable al maestro, y registrar el experimento. Si pasa → E5
> (exportación ONNX + INT8 + benchmark en el TECNO, RNF-01/02/03). Si falla →
> conforme al punto 4 del ADR 0009 y al ADR 0010 no se pagan más iteraciones
> del mismo mecanismo: la decisión vuelve al usuario (opción (c), servidor, o
> un ADR nuevo). Sólo destruir la instancia tras traer los artefactos con
> valor.

## 2. Repositorio y entorno

- Workspace: `/Users/davidsilva/VisualStudioCodeProjects/Avatar`.
- Rama: `main`.
- Remote: `origin`, repositorio GitHub `dasilvabalautaro/Avatar`, por SSH.
- El agente gestiona commits y push; nunca commitear `keys-git.md` ni
  credenciales.
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

- pytest: 71 pruebas superadas;
- Ruff: sin hallazgos;
- mypy estricto: sin errores en 41 archivos fuente;
- Gradle `:app:assembleDebug`: exitoso;
- smoke dataset: 64 muestras, 64 hashes únicos, cero hallazgos (generador v3,
  manifiesto v1.1.0);
- release v2.0.0 de entrenamiento congelada, lock SHA-256
  `79ecdd3f36301c4462372be35e93f66cee3e52f51d6992050728da8dc84334a2`;
- release v2.1.0 de entrenamiento congelada (1024 muestras), manifiesto
  SHA-256 `8e54942ef99711eb9c9ef80d2d33611168fc7480024c42b668bd2f62f6d91b5d`
  y paquete `transfer/avatarface-training-procedural-v2-1.tar`
  (5,179,904 bytes, SHA-256
  `f13d2cb4f8b113c9fd28d70ed265745b75167ee6694140980d0c37ff87afc37a`);
- release v2.2.0 de entrenamiento congelada (1024 muestras nativas de 512 px),
  manifiesto SHA-256
  `53fa3b374dc1ed48e59a4510b2348fd43ef6c12027420ccc0fa69e4f4a2616f8` y paquete
  `transfer/avatarface-training-procedural-v2-2.tar` (7,171,584 bytes, SHA-256
  `47744cf589e1160c50d22e852cf7886caebfa2c8fef50ea49f0b66a9c0026fd6`);
- checkpoint LoRA escala-4 (500 pasos, `lr=5e-5`, 512 px, rango 32,
  atención+FFN, sobre v2.2.0) verificado en local: SHA-256
  `b069f9bbe5651088ddeff71f5be4b496e4840d9985d96560c4faa1c4c7cbc4e2`;
- no quedó ningún proceso `com.avatarface.app` activo en el teléfono;
- la RTX 4090 de Vast quedó a 0 % de utilización y 1 MiB de memoria ocupada.

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
  `1c2f7aa82164ee9f07987b707d12b992f4a1f7a9803ff50daf0f120cdcefedba`;
- tamaño aproximado: 309 KiB;
- 64 hashes de imagen únicos;
- captions con plantilla «flat vector avatar face of an adult, …» (RF-09), con
  detalle de forma de ojos y accesorios ampliado (generador v3, v1.1.0);
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

Las antiguas tareas P0 (loader reproducible, microentrenamiento local
reanudable, regresión congelada y licencias) están completadas. También están
completadas la restauración de pesos en Vast, la compuerta base-only y la
repetición del piloto LoRA (ver `docs/experiments/wuerstchen-base-only-2026-08-15.md`
y `docs/experiments/wuerstchen-lora-pilot-v3-2026-08-15.md`). Completadas
además el diseño de escala-1, el ADR 0007 y el filtro de sólo adultos
(P2, puntos 1–3; el punto 4 queda incorporado a la lista de verificación de
`docs/lora-scale-1-design.md`).

### P1. Experimentos LoRA escala-1, escala-2, escala-3 y escala-4

Completados (2026-08-15, 2026-08-16 y 2026-08-17). Escala-2 demostró que más
pasos con la release v2.0.0 ya no mejoran la fidelidad de atributos; escala-3
demostró que tampoco la mejora del dataset (v2.1.0, doble de muestras y
captions más finos) destraba color de ojos, pecas ni pendientes; escala-4
demostró que tampoco lo destraban la resolución nativa de 512 px (v2.2.0) ni
el LoRA ampliado (rango 32, atención+FFN): el techo está en el modelo base
para estos rasgos finos. Ver `docs/experiments/wuerstchen-lora-scale-2-2026-08-16.md`,
`docs/experiments/wuerstchen-lora-scale-3-2026-08-17.md` y
`docs/experiments/wuerstchen-lora-scale-4-2026-08-17.md`. El ciclo de escalado
LoRA queda cerrado; la siguiente fase es integración con destilación (P2).

### P2. Brecha receta oficial vs. presupuesto móvil

Cerrada como decisión documentada (ADR 0007): la receta oficial queda para
GPU; la integración móvil exige destilación o reducción de pasos, que requerirá
su propio ADR antes de cualquier benchmark del modelo real en el dispositivo.

### P3. Dataset futuro

Release v2.1.0 completada (2026-08-16): el generador v3 usa la plantilla
«of an adult» con detalle de forma de ojos (`eye_shape`) y accesorios ampliados
(gafas cuadradas, gafas de sol), y la release duplica las muestras (1024).
Quedó congelada con nuevo lock SHA-256 y empaquetada para la ruta directa.
La corrida LoRA de 500 pasos sobre v2.1.0 y su comparación con escala-2 están
completadas (2026-08-17). La decisión del usuario quedó registrada el mismo
día: experimento escala-4 con resolución 512 y LoRA ampliado sobre la release
v2.2.0; ver «Próxima tarea exacta» en la sección 1 y
`docs/lora-scale-4-design.md`.

## 10. Puntos de vigilancia activos

- **Cuota pública de Drive:** sigue agotada para el paquete Stage B; la vía
  que funcionó es la descarga autenticada con rclone/OAuth de la cuenta
  propietaria. El token sólo existe en la instancia; no copiarlo a Git.
  Los 18.70 GB parciales de la descarga antigua por rangos están descartados
  y nunca deben usarse para entrenamiento.
- **Pesos en Vast:** la vía de descarga directa verificada quedó probada en la
  sesión de escala-1 (33/33 archivos, SHA-256 completo). Si la instancia se
  destruye, la re-entrada completa la hace `scripts/bootstrap-vast.sh` (ver
  «Próxima tarea exacta»). El paquete local de contingencia
  `transfer/avatarface-wuerstchen-v2-trimmed-20260815.tar` (24.1 GB) y los
  checksums huérfanos de paquetes antiguos se borraron el 2026-08-15. En
  `transfer/` conviven los `.tar` históricos de v2.0.0/smoke v1
  (`avatarface-training-procedural-v2-dataset.tar`, `avatarface-smoke-dataset.tar`,
  ya obsoletos y reconstruibles) con los paquetes vigentes de las releases
  v2.1.0 y v2.2.0 (`avatarface-training-procedural-v2-1.tar`,
  `avatarface-training-procedural-v2-2.tar`,
  `avatarface-smoke-procedural.tar`),
  su `SHA256SUMS`, el manifiesto recortado y el README.
- **Validaciones históricas inválidas:** la validación anterior (4 timesteps,
  256 px, guía por defecto) era inválida; no recuperar conclusiones de calidad
  de las muestras de `lora-pilot-v2` ni de `lora-pilot-v2-lr1e5`.
- **Costo GPU:** detener la instancia Vast al terminar cada tarea y verificar
  0 % de utilización al cerrar cada sesión remota.
- **Filtro de menores:** vigilar que prompts, captions del dataset y
  evaluación visual apliquen la restricción de sólo adultos de forma
  consistente.

## 11. Riesgos y prohibiciones vigentes

- **Prohibido generar, entrenar o validar avatares de menores de edad**; el
  producto es sólo para adultos (RF-09, riesgo R-16).
- No usar modelos, encoders, datasets o pesos con restricciones de uso.
- No asumir que la licencia del repositorio cubre todos los componentes.
- No usar rostros reales en el smoke dataset.
- No aceptar métricas de emulador como métricas finales.
- No ejecutar ADB sin `-s 14254155BM000874`.
- No volver a incluir checksum/posproceso en la latencia de inferencia.
- No usar NNAPI en el baseline actual.
- No desplegar `target` todavía.
- No commitear `keys-git.md` ni credenciales de ningún tipo.
- No conservar descargas reconstruibles.
- Flujo de transferencia obligatorio: máquina local → `.tar` + SHA256SUMS →
  Drive → descarga en Vast; nunca descargar directamente desde Vast.ai y
  verificar SHA-256 antes y después de cada transferencia. **Excepción
  (2026-08-15):** los pesos públicos de HuggingFace fijados por hash en
  `model-manifest.json` pueden descargarse directamente en la instancia con
  `scripts/download-wuerstchen-weights.py` + verificación completa de
  `scripts/verify-model-manifest.py` (ver `transfer/README.md`). Los datasets
  propios siguen el flujo estándar.

## 12. Primeros comandos de la próxima sesión

```bash
cd /Users/davidsilva/VisualStudioCodeProjects/Avatar
sed -n '1,140p' docs/HANDOFF.md
git status --short
git log -1 --oneline
.venv/bin/pytest
.venv/bin/avatar-face audit-dataset \
  --manifest data/smoke-procedural/manifest.json
```

Después, retomar desde la sección 9 (P1 cerrada con escala-3; la dirección
siguiente es una decisión del usuario — ver «En curso» y «Próxima tarea
exacta» en la sección 1).
