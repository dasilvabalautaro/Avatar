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

## Pendiente para cerrar la fase

- Confirmación del titular antes de publicar la dedicación CC0.
- Definir y aplicar licencia al código completo del repositorio.
- Diseñar muestreo estratificado para balance exacto.
- Ampliar diversidad visual y taxonomía de atributos.
- Incorporar un dataset de calidad mayor manteniendo 100 % de procedencia.
- Detectar similitud perceptual, además de duplicados exactos.
- Crear un loader y ejecutar microentrenamiento local.

## Siguiente compuerta

Implementar el loader reproducible y un autoencoder/decoder smoke entrenable para
verificar batching, pérdida, checkpoint y reanudación con estas 64 muestras antes
de usar Vast.ai.
