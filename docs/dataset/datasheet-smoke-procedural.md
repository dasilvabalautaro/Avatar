# Datasheet: AvatarFace Smoke Procedural v2

## Resumen

Dataset pequeño para validar carga de imágenes, captions, splits, entrenamiento y
reanudar checkpoints. No representa el dataset final ni sirve para evaluar
calidad, realismo o equidad del modelo.

## Creación

- Generador: `avatarface-procedural-v2`.
- Resolución: 256 × 256 RGB PNG.
- Muestras: 64.
- Train: 50; validation: 7; test: 7.
- Seed raíz: 42.
- Tamaño local total: aproximadamente 316 KiB.
- Manifiesto SHA-256:
  `146026fb9c4e99ed92ac7ba359b35ff7c9aee69c5f9fd80f29874cac672b7ae2`.

Las imágenes se dibujan mediante primitivas geométricas propias. No se usan
fotografías, fuentes, modelos generativos, assets descargados ni identidades
reales.

## Campos auditados

Cada muestra registra identificador, ruta relativa, caption, atributos, fuente,
creador, licencia, URL de licencia, estado de consentimiento, SHA-256, split y
bandera sintética.

El comando de auditoría recalcula hashes y rechaza rutas inseguras, archivos
ausentes, identificadores o imágenes duplicadas, licencias no aprobadas y
muestras que no sean sintéticas. También compara una firma perceptual RGB de
miniaturas 16 × 16 para detectar similitud no deseada.

## Cobertura actual

- Seis categorías de tono de piel estilizado: 10–11 ejemplos cada una.
- Cuatro estilos de cabello, expresiones, accesorios y formas faciales: 16 ejemplos cada uno.
- Tres variantes de cejas y nariz: 21–22 ejemplos cada una.
- 64 hashes de imagen únicos.
- Sin pares perceptualmente similares según la auditoría.

El muestreo es estratificado y la diferencia de cada categoría primaria no
supera una muestra. La v2 añade formas faciales, cejas y narices.

## Licencia y derechos

AvatarFace es el creador de las imágenes y metadatos procedimentales. El titular
del proyecto confirmó su dedicación a CC0-1.0 el 2026-08-13. La declaración,
su alcance y sus exclusiones están en `docs/dataset/CC0-DEDICATION.md`.

CC0 no elimina derechos de privacidad o imagen de terceras personas; este dataset
evita ese riesgo porque no representa personas reales. Esta documentación no
constituye asesoría legal.

La regresión de generación está congelada en
`configs/regression-fixtures.json`: fija el generador v3, la seed raíz 42, hash
del manifiesto y captions representativos con sus hashes. No se actualizará por
regeneración accidental; todo cambio exige revisar su causa y este datasheet.
La plantilla vigente es «flat vector avatar face of an adult, …» con detalle de
forma de ojos y accesorios ampliado (generador v3, release de entrenamiento
v2.1.0).

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
