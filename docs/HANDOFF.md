# Handoff de AvatarFace

Actualizado: 2026-08-16, zona horaria `America/La_Paz`.

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

En curso:

- Fase 2 — siguiente paso: repetir el escalado LoRA sobre la release **v2.1.0**
  (500 pasos, `lr=5e-5`, mismos 8 prompts congelados de evaluación) y comparar
  contra escala-2. La instancia de escala-2 quedó encendida a la espera de
  decisión del usuario (apagar o reutilizar); GPU verificada al 0 %.

Requisito rector nuevo (2026-08-15): el producto genera **sólo rostros de
adultos**; está prohibido generar avatares de menores de edad. Ver RF-09 en
`docs/product-requirements.md`, riesgo R-16 y la sección 11.

Próxima tarea exacta:

> Entrenar y evaluar el escalado LoRA sobre la release v2.1.0 (ya congelada):
> 1. Transferir `transfer/avatarface-training-procedural-v2-1.tar` y
>    `transfer/avatarface-smoke-procedural.tar` con su `SHA256SUMS` por la ruta
>    directa de `transfer/README.md` (ambos < 100 MB).
> 2. En una GPU Vast (re-entrada con `scripts/bootstrap-vast.sh`, ya probada en
>    escala-2 y actualizada a v2.1): entrenar 500 pasos con `lr=5e-5` y evaluar
>    con `scripts/eval-visual-lora.sh` (mismo conjunto congelado de 8 prompts).
> 3. Comparar contra escala-2 con la lista del diseño (incluye edad aparente
>    adulta, RF-09), respaldo local con SHA-256, doc en `docs/experiments/`,
>    GPU al 0 % y avisar al usuario para apagar la instancia.

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

- pytest: 52 pruebas superadas;
- Ruff: sin hallazgos;
- mypy estricto: sin errores en 38 archivos fuente;
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

### P1. Experimentos LoRA escala-1 y escala-2

Completados (2026-08-15 y 2026-08-16). Escala-2 demostró que más pasos con la
release v2.0.0 ya no mejoran la fidelidad de atributos; ver
`docs/experiments/wuerstchen-lora-scale-2-2026-08-16.md`. La re-entrada
automatizada con `scripts/bootstrap-vast.sh` quedó probada de extremo a
extremo (incluye la copia de `SHA256SUMS` que exige el preflight).

### P2. Brecha receta oficial vs. presupuesto móvil

Cerrada como decisión documentada (ADR 0007): la receta oficial queda para
GPU; la integración móvil exige destilación o reducción de pasos, que requerirá
su propio ADR antes de cualquier benchmark del modelo real en el dispositivo.

### P3. Dataset futuro

Release v2.1.0 completada (2026-08-16): el generador v3 usa la plantilla
«of an adult» con detalle de forma de ojos (`eye_shape`) y accesorios ampliados
(gafas cuadradas, gafas de sol), y la release duplica las muestras (1024).
Quedó congelada con nuevo lock SHA-256 y empaquetada para la ruta directa.
Pendiente: la corrida LoRA de 500 pasos sobre v2.1.0 y su comparación con
escala-2 (ver «Próxima tarea exacta» en la sección 1).

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
  ya obsoletos y reconstruibles) con los paquetes vigentes de la release v2.1.0
  (`avatarface-training-procedural-v2-1.tar`, `avatarface-smoke-procedural.tar`),
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

Después, retomar desde la sección 9 (P3: la release v2.1.0 ya está congelada y
empaquetada; falta la repetición del escalado LoRA sobre ella — ver «Próxima
tarea exacta» en la sección 1).
