# Piloto LoRA Würstchen v3 — 2026-08-15

Repetición del piloto de 20 pasos sobre la release `training-procedural-v2`
autorizada tras superar la compuerta `base-only`
(`docs/experiments/wuerstchen-base-only-2026-08-15.md`). Misma instancia Vast
(RTX 4090; IP y puerto registrados fuera de Git).

## Configuración

- Script: `scripts/run_wuerstchen_lora_pilot.py` con el loader de texto
  corregido (ver el documento de la compuerta base-only).
- LoRA: rango 16, alpha 16, dropout 0.05; sólo `to_q`, `to_k`, `to_v`,
  `to_out.0`. 6,291,456 parámetros entrenables.
- Semilla 42, resolución 256, batch 1, `lr=1e-5` (por defecto), dtype bf16.
- Dataset: release v2.0.0, manifiesto verificado en la instancia con lock
  SHA-256 `79ecdd3f36301c4462372be35e93f66cee3e52f51d6992050728da8dc84334a2`.

Pérdidas por paso: `0.000916, 0.238580, 0.034005, 0.001902, 0.076002,
0.063435, 0.052501, 0.138142, 0.032994, 0.093703, 0.182703, 0.333869,
0.196840, 0.012912, 0.947406, 0.193307, 0.164005, 0.114536, 0.357111,
0.690381`.

## Validación con el validador corregido

`validate_wuerstchen_lora.py --checkpoint pilot-checkpoint.pt` con la receta
oficial (30 timesteps, fp16, 1024, guía 8.0, prompt negativo, semilla 42):

```text
prior_ok shape=(1, 16, 24, 24) min=-171.625000 max=186.375000 \
  mean=-5.582031 std=53.531250
validation_ok size=(1024, 1024) pixel_mean=108.392174 pixel_std=107.738274
```

## Inspección visual y respaldo

La muestra `validation.png` es un **rostro de avatar válido** (persona adulta,
estilo vectorial plano, cabello negro, ojos verdes, fondo cian), sin ruido ni
mosaico; muy próxima a la salida `base-only`, lo esperado tras sólo 20 pasos
con `lr=1e-5`.

Respaldos locales:

- `artifacts/lora-pilot-v3/pilot-checkpoint.pt`, SHA-256
  `17650299c27613d2f51a16035f16231cc1658d1915d7e0ff87b0b1bab8e2fb9f`.
- `artifacts/lora-pilot-v3/validation.png`, 1024 × 1024, SHA-256
  `2653ed664e6c1c64d5caf2bd01661a344fa6db37fbb2d7308e09acf77c8c1ba4`.

**Conclusión:** la integración LoRA completa (entrenamiento → checkpoint →
validación con receta oficial) queda verificada de extremo a extremo. Las
muestras de `lora-pilot-v2` y `lora-pilot-v2-lr1e5` siguen invalidadas por la
configuración anterior del validador; no se recuperan conclusiones de ellas.

La GPU quedó a `0 %` de utilización y `1 MiB` de memoria ocupada al cerrar la
sesión. La instancia se detiene desde la consola de Vast (sin CLI local).
