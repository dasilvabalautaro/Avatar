# Estado de la Fase 2 — dataset legal y auditable

## Resultado actual

La infraestructura mínima y un smoke dataset procedimental fueron creados. La
compuerta automatizada aprueba integridad y política técnica: 64 muestras, 64
hashes únicos y cero hallazgos.

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

## Pendiente para cerrar la fase

- Incorporar un dataset de calidad mayor manteniendo 100 % de procedencia.

## Siguiente compuerta

Las captions/prompts/seeds están congelados y las licencias de publicación están
formalizadas. Antes de entrenamiento serio queda ampliar el dataset manteniendo
100 % de procedencia; después puede prepararse el preflight de Vast.ai.

## Regresión congelada

`configs/regression-fixtures.json` fija el generador v2, 64 muestras, seed raíz
42, hash del manifiesto, ocho captions representativos con sus hashes y cuatro
prompts con seeds límite. `tests/test_regression_fixtures.py` regenera el
dataset y rechaza cualquier deriva no declarada.
