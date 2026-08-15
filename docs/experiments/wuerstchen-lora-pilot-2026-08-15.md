# Piloto LoRA Würstchen — 2026-08-15

Se ejecutó un piloto de 20 pasos sobre `training-procedural-v2` en la RTX 4090
de Vast (dirección IP y puerto SSH registrados fuera de Git). El paso previo
forward/backward también pasó (`loss=0.009686`).

## Configuración

- Prior: `warp-ai/wuerstchen-prior-model-base`, revisión fijada en el manifiesto.
- Encoder Stage B: `dome272/wuerstchen/model_v2_stage_b.pt`, SHA-256
  `aba31f166a8dfb672fc81b63336b12e4667a10ddfb783a822b8fe20b356a899c`.
- LoRA: rango 16, alpha 16, dropout 0.05; sólo atención `to_q`, `to_k`, `to_v`
  y `to_out.0`.
- Semilla: 42; resolución: 256; batch: 1; learning rate: `1e-4`; dtype: bf16.
- Dataset: release v2.0.0, lock SHA-256
  `79ecdd3f36301c4462372be35e93f66cee3e52f51d6992050728da8dc84334a2`.

Pérdidas por paso: `0.009686, 4.032487, 0.153856, 0.009437, 0.534020,
0.166763, 0.118438, 0.326023, 0.049946, 0.146853, 1.102002, 2.459612,
1.026946, 0.020465, 4.080022, 0.935848, 0.835392, 0.229097, 2.849503,
3.490827`.

## Resultado y respaldo

El checkpoint se respaldó localmente en
`artifacts/lora-pilot-v2/pilot-checkpoint.pt` (12 MB), SHA-256
`5ccf0084d4bedf82bc998ee6234b478f9b667d0e6b4a43e288412927a203cab2`.

Es un piloto de compatibilidad y trazabilidad, no una evaluación de calidad ni
un entrenamiento de producción; las pérdidas fluctúan y requieren validación
visual antes de escalar.

La inferencia posterior sí completó y produjo `artifacts/lora-pilot-v2/validation.png`
(256 × 256, SHA-256
`5b3b4523b7cea35d8fc3341ede56ff894e08964132029374a8a951bf94b8544d`), pero la
inspección visual mostró una imagen prácticamente en blanco. El resultado de
calidad es, por tanto, **fallido para escalar**; el checkpoint sólo conserva
valor como prueba de integración, no como modelo utilizable.

## Repetición controlada

Se repitió exactamente el mismo piloto con `learning rate=1e-5` para descartar
que `1e-4` fuese el único problema. El segundo checkpoint se respaldó en
`artifacts/lora-pilot-v2-lr1e5/pilot-checkpoint.pt`, SHA-256
`04edb8435147997e202c84c6a3d53fb90cba33498f58d3b2fe9cd93638c51bd0`.

La muestra correspondiente está en
`artifacts/lora-pilot-v2-lr1e5/validation.png`, 256 × 256, SHA-256
`6ee257d2625343b8937dd6c9535f13262d4bb21175cefa9471b87f408336d408`.
La imagen conserva bandas/mosaico púrpura y tampoco es utilizable. La muestra
`base-only` ya presentaba el mismo tipo de salida inválida; por ello el bloqueo
actual se clasifica como **incompatibilidad de inferencia/decodificador aún no
resuelta**, no como motivo para aumentar pasos o pagar una corrida larga.

**Decisión:** detener la GPU después de conservar los respaldos. Antes de otro
entrenamiento se debe corregir y probar la ruta prior → decoder con una muestra
base válida, y repetir sólo el smoke/piloto cuando esa compuerta pase.

La comprobación final en Vast mostró RTX 4090 a `0 %` de utilización y `1 MiB`
de memoria ocupada; no quedó ningún proceso de entrenamiento activo.

## Diagnóstico posterior

La conclusión visual anterior no permite juzgar los checkpoints: el validador
usaba sólo 4 pasos uniformes del prior, resolución 256 y la guía predeterminada.
La receta oficial para el checkpoint base usa la secuencia
`DEFAULT_STAGE_C_TIMESTEPS` de 30 puntos, `float16`, resolución 1024 y
`guidance_scale=8.0` con prompt negativo. El propio model card indica que el
checkpoint base fue entrenado para 1024–1536 y necesita una guía mayor.

Se corrigió `scripts/validate_wuerstchen_lora.py` para reproducir esos valores,
cargar el scheduler del decoder desde su configuración congelada, usar una
semilla explícita, rechazar embeddings no finitos y registrar estadísticas de
la imagen. La carga del encoder de texto ahora es estricta: cualquier peso
faltante o inesperado aborta la prueba en lugar de continuar silenciosamente.

Fuentes de referencia:

- [Model card de `wuerstchen-prior-model-base`](https://huggingface.co/warp-ai/wuerstchen-prior-model-base)
- [Script oficial de entrenamiento Diffusers v0.31.0](https://github.com/huggingface/diffusers/blob/v0.31.0/examples/wuerstchen/text_to_image/train_text_to_image_prior.py)

La clasificación actual es **validación anterior inválida por configuración**.
Hace falta una nueva ejecución `base-only` en GPU con el validador corregido;
no se repetirá LoRA hasta que esa imagen base pase inspección visual.

## Nueva instancia y bloqueo de restauración

La nueva instancia (ID, IP y puerto registrados fuera de Git) resultó ser un
contenedor nuevo, sin volumen persistente. Se clonó el repositorio, se
instalaron las dependencias fijadas y se restauró por transferencia directa el
dataset de 2.5 MB. La RTX 4090 permaneció sin carga de entrenamiento.

Google Drive rechazó por cuota tanto el paquete Stage B de 34.3 GB como el
paquete anterior de 30 GB (IDs de Drive registrados fuera de Git). Sus
archivos `SHA256SUMS` sí se descargaron y conservaron los hashes esperados. No
se sustituyó la ruta acordada por una transferencia local: hace falta crear en
Drive una copia con ID nuevo y reanudar la descarga desde allí.

La copia de Drive heredó la misma cuota. Un intento reanudable por rangos llegó
a 18.70 GB, pero el endpoint comenzó a devolver la página de cuota incluso para
bloques pequeños; se detuvo sin usar los datos parciales para entrenamiento. Se
preparó localmente un paquete nuevo, con bytes y nombre diferentes:

- `transfer/avatarface-wuerstchen-v2-stageb-fresh-20260815.tar`
- Tamaño: 34,309,578,752 bytes.
- SHA-256: `c25196411187496755f8c1001a5ce90aa344aa32adabcf26c988d2a8d0a7a92a`.
- Checksum: `transfer/SHA256SUMS-stageb-fresh-20260815`.

El listado auditó 51 entradas, todas bajo `models/wuerstchen-v2/`, e incluye
`model-manifest.json`. Este archivo debe subirse realmente a Drive; una copia
servidor-a-servidor del paquete anterior no obtiene una cuota independiente.
