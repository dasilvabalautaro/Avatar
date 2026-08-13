# Datasheet: AvatarFace Smoke Procedural v1

## Resumen

Dataset pequeño para validar carga de imágenes, captions, splits, entrenamiento y
reanudar checkpoints. No representa el dataset final ni sirve para evaluar
calidad, realismo o equidad del modelo.

## Creación

- Generador: `avatarface-procedural-v1`.
- Resolución: 256 × 256 RGB PNG.
- Muestras: 64.
- Train: 50; validation: 7; test: 7.
- Seed raíz: 42.
- Tamaño local total: aproximadamente 316 KiB.
- Manifiesto SHA-256:
  `0cc3ecb993288b86f17e547411caef860b19e09669e1eac1020e927c0b8aae7c`.

Las imágenes se dibujan mediante primitivas geométricas propias. No se usan
fotografías, fuentes, modelos generativos, assets descargados ni identidades
reales.

## Campos auditados

Cada muestra registra identificador, ruta relativa, caption, atributos, fuente,
creador, licencia, URL de licencia, estado de consentimiento, SHA-256, split y
bandera sintética.

El comando de auditoría recalcula hashes y rechaza rutas inseguras, archivos
ausentes, identificadores o imágenes duplicadas, licencias no aprobadas y
muestras que no sean sintéticas.

## Cobertura actual

- Seis categorías de tono de piel estilizado: 8–16 ejemplos cada una.
- Cuatro estilos de cabello: 10–22 ejemplos cada uno.
- Cuatro expresiones: 13–22 ejemplos cada una.
- Cuatro estados de accesorios: 11–21 ejemplos cada uno.
- 64 hashes de imagen únicos.

La distribución no está perfectamente balanceada. En la siguiente versión el
muestreo deberá ser estratificado, no aleatorio, y se ampliarán formas faciales,
peinados, edades aparentes, cejas, narices y accesorios.

## Licencia y derechos

AvatarFace es el creador de las imágenes y metadatos procedimentales. El
manifiesto declara la intención de distribuirlos bajo CC0-1.0. Antes de publicar
el dataset, el titular del proyecto debe confirmar expresamente la dedicación.

CC0 no elimina derechos de privacidad o imagen de terceras personas; este dataset
evita ese riesgo porque no representa personas reales. Esta documentación no
constituye asesoría legal.

## Limitaciones

- El estilo es plano y geométrico.
- No hay iluminación, poses, fondos complejos ni oclusiones importantes.
- Las categorías son descriptores artísticos, no afirmaciones demográficas.
- Los captions están en inglés y siguen una plantilla.
- No se ha realizado evaluación perceptual ni de similitud semántica.

## Reproducción

```bash
.venv/bin/avatar-face generate-smoke-dataset \
  --output-dir data/smoke-procedural \
  --samples 64 \
  --seed 42 \
  --overwrite

.venv/bin/avatar-face audit-dataset \
  --manifest data/smoke-procedural/manifest.json
```
