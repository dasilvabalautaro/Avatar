# Estado de la Fase 2 — dataset legal y auditable

## Resultado actual

El smoke dataset queda como evidencia histórica del flujo. La siguiente release
se genera con una fuente especializada aprobada, un manifiesto v2 y un lock de
release que fija hashes de manifiesto, imágenes y splits.

## Completado

- Entidad de muestra con procedencia, licencia, consentimiento, hash y split.
- Generador determinista de avatares 256 × 256 sin activos externos.
- Manifiesto JSON reproducible con captions y atributos.
- Splits train/validation/test.
- Auditoría independiente de archivos, hashes, duplicados y licencia.
- Datasheet y matriz de licencias.
- Pillow fijado en `requirements-dataset.lock` e instalado sin caché de descarga.
- Inspección visual de muestras y auditoría de cobertura.
- Loader PyTorch determinista, exclusivo del manifiesto y normalizado a `[-1, 1]`.
- Preflight de hashes de todas las muestras y verificación opcional al abrirlas.
- Autoencoder local mínimo con checkpoint, reconstrucción de validation y reanudación.
- Muestreo estratificado con balance exacto o diferencia máxima de una muestra.
- Diversidad geométrica: forma facial, cejas y nariz, además de color y accesorios.
- Detección de similitud perceptual RGB, además de duplicados exactos.

## Fuente aprobada

`AF-PROC-001` (avatares procedimentales propios, CC0) es la única fuente
aprobada para ingestión. Las fuentes por encargo y aportes con consentimiento
son condicionales y no pueden entrar sin evidencia por activo. La política y el
registro están en `docs/dataset/source-approval.md` y
`configs/dataset-sources.json`.

## Ejecución de la release ampliada

```bash
.venv/bin/avatar-face generate-training-dataset \
  --output-dir data/training-procedural-v2-1 --samples 1024 --seed 42
.venv/bin/avatar-face audit-dataset \
  --manifest data/training-procedural-v2-1/manifest.json
.venv/bin/avatar-face freeze-dataset \
  --manifest data/training-procedural-v2-1/manifest.json \
  --version v2.1.0
.venv/bin/avatar-face train-smoke \
  --manifest data/training-procedural-v2-1/manifest.json \
  --output-dir artifacts/training-procedural-v2-1 --steps 5
```

## Release local congelada v2.1.0

Se generó y verificó localmente `v2.1.0` el 2026-08-16, con el generador
`avatarface-procedural-v3`: la plantilla de captions «of an adult» ahora detalla
la forma de los ojos (`almond`, `round`, `narrow`) y amplía los accesorios
(gafas cuadradas y gafas de sol), respondiendo al límite de fidelidad observado
en el experimento de escala-2:

- 1024 PNG; splits `train=818`, `validation=103`, `test=103`;
- manifiesto SHA-256:
  `8e54942ef99711eb9c9ef80d2d33611168fc7480024c42b668bd2f62f6d91b5d`;
- auditoría: 1024 hashes únicos, cero duplicados/similitudes y cero hallazgos;
- lock: `data/training-procedural-v2-1/dataset-v2.1.0.lock.json`;
- paquete: `transfer/avatarface-training-procedural-v2-1.tar`, SHA-256
  `f13d2cb4f8b113c9fd28d70ed265745b75167ee6694140980d0c37ff87afc37a`,
  5,179,904 bytes; por la regla de transporte (≤100 MB), irá directamente de la
  máquina local a Vast.ai, sin Drive.

Antes de empaquetar o transferir, y después de descargar en Vast, ejecutar:

```bash
.venv/bin/avatar-face verify-frozen-dataset \
  --manifest data/training-procedural-v2-1/manifest.json \
  --lock data/training-procedural-v2-1/dataset-v2.1.0.lock.json
```

## Release v2.0.0 (histórica)

Se generó y verificó localmente `v2.0.0` el 2026-08-13:

- 512 PNG; splits `train=408`, `validation=52`, `test=52`;
- manifiesto SHA-256:
  `79ecdd3f36301c4462372be35e93f66cee3e52f51d6992050728da8dc84334a2`;
- auditoría: 512 hashes únicos, cero duplicados/similitudes y cero hallazgos;
- lock: `data/training-procedural-v2/dataset-v2.0.0.lock.json`;
- microtraining local: 5 pasos, pérdida final `0.35704541206359863`, checkpoint
  asociado al hash anterior.
- paquete: `transfer/avatarface-training-procedural-v2-dataset.tar`, SHA-256
  `d8fd1b2d284a6fe0c4f77ad96d0869d53daa110709c02dd92301ae2da5907fc9`,
  2,577,920 bytes; por la regla de transporte (≤100 MB), irá directamente de la
  máquina local a Vast.ai, sin Drive.

Antes de empaquetar o transferir, y después de descargar en Vast, ejecutar:

```bash
.venv/bin/avatar-face verify-frozen-dataset \
  --manifest data/training-procedural-v2/manifest.json \
  --lock data/training-procedural-v2/dataset-v2.0.0.lock.json
```

## Smoke remoto y respaldo

El 2026-08-13 se completó la repetición en Vast.ai por transferencia directa
(el `.tar` mide menos de 100 MB):

- restauración del paquete y verificación del lock: aprobadas;
- preflight: RTX 4090, 47.37 GiB VRAM visibles por PyTorch, 127.86 GiB libres;
- smoke training: 5 pasos, pérdida final `0.35704541206359863`, manifiesto
  `79ecdd3f36301c4462372be35e93f66cee3e52f51d6992050728da8dc84334a2`;
- respaldo local del checkpoint: SHA-256
  `69f428fc71b6c77da5ecc25c5c769109ea098877ad57056fd41b50eb5d4b496f`;
- respaldo local de la reconstrucción: SHA-256
  `a970d41c4f33c6a92f3608db25847a817823e95543a1551fc8d9460c5b428127`.

El microentrenador es deliberadamente CPU para comprobar el flujo de datos,
checkpoint y restauración; la GPU fue validada por el preflight, no empleada por
este autoencoder de humo.

## Siguiente decisión autorizada

La compuerta operativa local → Vast → checkpoint → respaldo está superada. Ya
puede seleccionarse un modelo base permisivo y redactarse el plan de LoRA/
fine-tuning en GPU. La ampliación futura con fuentes condicionales se evaluará
por separado, con evidencia por activo.

## Siguiente compuerta

La selección de modelo base y el plan LoRA/fine-tuning en GPU permanecen
bloqueados hasta que se congele una release ampliada y se repita el smoke
training completo sobre ella.

## Regresión congelada

`configs/regression-fixtures.json` fija el generador v3, 64 muestras, seed raíz
42, hash del manifiesto, ocho captions representativos con sus hashes y cuatro
prompts con seeds límite. `tests/test_regression_fixtures.py` regenera el
dataset y rechaza cualquier deriva no declarada.
