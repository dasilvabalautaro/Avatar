# Compuerta base-only Würstchen — 2026-08-15

Segunda sesión del 2026-08-15 sobre una instancia nueva de Vast (RTX 4090;
dirección IP y puerto SSH registrados fuera de Git). Objetivo: restaurar los
pesos Stage B y ejecutar la compuerta `base-only` con el validador corregido,
según `docs/HANDOFF.md` sección 9.

## Restauración de pesos

Google Drive mantenía la cuota pública de descarga agotada para el paquete
fresh (respuesta "Quota exceeded" por rangos, incluso desde la IP de la
instancia). Se resolvió sin cambiar la ruta acordada usando la **API
autenticada de Drive** mediante rclone con OAuth de la cuenta propietaria
(configuración y token sólo en la instancia, nunca en Git). La descarga
Drive → Vast completó los 34,309,578,752 bytes y el SHA-256 completo coincidió:

- SHA-256: `c25196411187496755f8c1001a5ce90aa344aa32adabcf26c988d2a8d0a7a92a`.
- Manifiesto: `model_manifest_ok components=4 files=34 bytes=34309480546`,
  `manifest_sha256=0b1c6b0ee3efdf4d1e5985e6728b962f9c996b7cd9f4bf71190c561959fe660a`.

El `.tar` remoto se eliminó tras verificar. Los pesos quedaron en
`/workspace/models/wuerstchen-v2/`. El dataset `training-procedural-v2` se
restauró por transferencia directa (2.5 MB) y su manifiesto coincide con el
lock congelado `79ecdd3f36301c4462372be35e93f66cee3e52f51d6992050728da8dc84334a2`
(train 408, validation 52, test 52).

## Corrección del loader del encoder de texto

La primera ejecución del validador abortó en `load_text_model`
(`scripts/run_wuerstchen_lora_pilot.py`): los shards safetensors conservan el
prefijo `text_model.` del CLIP completo, pero `CLIPTextModel` de
`transformers==5.15.0` espera claves sin prefijo, y
`text_model.embeddings.position_ids` es un búfer no persistente que no forma
parte del `state_dict`. Se corrigió el loader para quitar el prefijo y excluir
`position_ids`, manteniendo la carga estricta. La copia usada en el piloto
original llevaba esta corrección de forma implícita; el commit anterior de los
scripts no la incluía.

## Ejecución de la compuerta

Comando: `validate_wuerstchen_lora.py --root /workspace --base-only` con la
receta oficial ya corregida: `DEFAULT_STAGE_C_TIMESTEPS` (30 puntos), fp16,
1024 × 1024, `guidance_scale=8.0`, prompt negativo, semilla 42, 12 pasos de
decoder.

Resultado:

```text
validation_ok output=/workspace/artifacts/validation/base-only-20260815.png \
  size=(1024, 1024) pixel_mean=108.027122 pixel_std=107.864189
```

Respaldo local: `artifacts/validation/base-only-20260815.png`, SHA-256
`aec14d50a43fead76b4a68e6622473f3b0825fe90e964c114e35da93c9173763`.

## Inspección visual

La imagen es un **rostro de avatar válido**: estilo vectorial plano, persona
adulta, cabello negro largo con raya al medio, ojos verdes, fondo cian/cielo,
anatomía coherente y sin ruido, bandas ni mosaico. Corresponde al prompt de la
receta y cumple la restricción de sólo adultos (RF-09).

**Decisión:** compuerta `base-only` **superada**. La ruta prior → decoder con
la receta oficial queda validada y se autoriza repetir el piloto LoRA de 20
pasos sobre la release v2 (P1 del HANDOFF).
