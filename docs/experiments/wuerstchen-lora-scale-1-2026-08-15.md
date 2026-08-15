# Experimento LoRA escala-1 — 2026-08-15

Primera corrida de escalado tras el piloto v3, ejecutada según
`docs/lora-scale-1-design.md` en una instancia Vast nueva (RTX 4090, 48 GiB;
IP y puerto registrados fuera de Git).

## Restauración del entorno (vía nueva)

La instancia anterior se descartó sin entrenar y el token rclone/OAuth murió
con ella. Con la excepción documentada en `transfer/README.md`, los pesos se
descargaron directamente en la instancia desde HuggingFace con
`scripts/download-wuerstchen-weights.py` y el manifiesto recortado
(`transfer/model-manifest-trimmed-20260815.json`, sin el duplicado
`open_clip_model.safetensors` de 10.2 GB):

```text
model_manifest_ok components=4 files=33 bytes=24151097654 \
  manifest_sha256=2dfcc73fa85dfd4afb419e02299b712d7cee77ac6b0f4911f139198e389fbd6f
```

Los 33 archivos verificaron SHA-256 contra el manifiesto. El dataset v2.0.0 se
transfirió por ruta directa (2.5 MB), restauración verificada con lock
`79ecdd3f36301c4462372be35e93f66cee3e52f51d6992050728da8dc84334a2` y auditoría
sin hallazgos. `preflight-vast` reportó `ready:true`.

## Entrenamiento

- Script: `scripts/run_wuerstchen_lora_pilot.py --steps 200 --learning-rate 5e-5`.
- LoRA: rango 16, alpha 16, dropout 0.05; sólo `to_q`, `to_k`, `to_v`,
  `to_out.0`. 6,291,456 parámetros entrenables.
- Semilla 42, resolución 256, batch 1, dtype bf16, 200 muestras distintas de
  train (de 408 disponibles en la release v2.0.0).
- Las primeras 20 pérdidas reproducen casi exactamente las del piloto v3
  (mismo orden de datos determinista): `0.000916, 0.238362, 0.034055, ...`
- Pérdida final (paso 200): `0.076270`. Sin inestabilidad con `lr=5e-5`
  (picos ocasionales < 1.0, igual que en el piloto).
- Checkpoint: `artifacts/lora-scale-1/pilot-checkpoint.pt`, SHA-256
  `7512290850151f72cdb06dd4801a0a6bbf96f37f4b36ade6126321913bc093a5`.

## Evaluación visual (receta oficial: 1024 px, 30 timesteps, guía 8.0, seed 42)

Los 8 prompts fijados del diseño (todos «of an adult», verificados con
`validate-prompt`) se generaron con el checkpoint LoRA y con `--base-only`.
Las 16 salidas pasaron la comprobación de degeneración (`pixel_std` sano) y la
inspección visual. Muestras en `artifacts/lora-scale-1/eval/` (SHA-256 abajo).

| # | Atributos clave del prompt | LoRA | base-only | Divergencia |
|---|---|---|---|---|
| 1 | happy, square, porcelain, side-parted black, green eyes, earrings, sky | adulto válido; happy, black hair, sky ✓; earrings/green ✗ | misma composición, rasgos distintos | sutil |
| 2 | confident, heart, light, bob brown, gray eyes, freckles, lavender | adulto válido; bob brown, lavender, gray eyes ✓; freckles ✗ | similar, con collar y diadema | sutil |
| 3 | calm, oval, deep, curly pink, blue eyes, round glasses, sky | adulto válido; **pink ✓**, glasses ✓ (rectangulares), sky ✓ | cabello púrpura ✗ | sutil; LoRA más fiel al color de cabello |
| 4 | smiling, round, brown skin, short blue, brown eyes, mint | adulto válido; **brown skin ✓**, smiling, mint ✓; cabello brown ✗ | piel más clara ✗ | sutil; LoRA más fiel al tono de piel |
| 5 | happy, heart, golden, bob blonde, green eyes, coral | adulto válido; coral, happy ✓; cabello verdoso ✗ | **blonde ✓** | clara; aquí base-only fue más fiel |
| 6 | calm, square, tan, curly auburn, gray eyes, earrings, sand | adulto válido; **auburn ✓**, calm ✓; fondo blanco ✗ | auburn parcial | sutil |
| 7 | confident, oval, brown skin, short black, brown eyes, lavender | adulto válido; **brown skin, black hair, lavender ✓** | piel más clara | sutil; LoRA más fiel |
| 8 | smiling, round, porcelain, side-parted pink, blue eyes, freckles, mint | adulto válido; pink, smiling, mint ✓; freckles/blue eyes ✗ | **blue eyes ✓**, contorno azul | sutil |

### Lista de verificación

1. **Rostro válido**: 16/16 sin ruido, mosaico ni imagen en blanco. ✓
2. **Atributos del prompt**: fidelidad parcial en ambos modelos — fondos y
   expresiones casi siempre correctos; color de ojos y accesorios débiles.
   El LoRA fue más fiel que la base en 3 casos (3, 4, 7) y menos fiel en 2
   (5, 8).
3. **Edad aparente adulta (RF-09)**: 16/16 representan adultos; ninguna
   muestra dudosa. ✓
4. **Divergencia respecto a base-only**: sutil pero real y consistente;
   200 pasos con `lr=5e-5` ya mueven el modelo sin degradarlo.

## Conclusión y siguiente paso

La muestra de escala-1 es **visualmente válida** según la compuerta del diseño.
El rango de lr queda acotado: `1e-5` estable pero casi sin efecto visible a 20
pasos; `5e-5` estable y con divergencia medible a 200 pasos; `1e-4` destruye la
salida. Se autoriza **escala-2**: 500–1000 pasos con `lr=5e-5` sobre las 408
muestras de train, con el mismo conjunto de evaluación, para comprobar si la
fidelidad de atributos (ojos, accesorios) mejora con más pasos.

## Hashes de las muestras (SHA-256)

```text
69e7395c2984562377f4a1103ee78b693b93ea0d48bcc3992c1d8b799e5f9df0  base-01.png
3586c4efa473105c6a73f450f77224ce968430d207b829f26c8a27edc3e31936  base-02.png
dca6598070a5efcd6984ea46deb27b12c2c1fd2121e3a118c4de6e329f17928d  base-03.png
10a739d2059890851d042701d69edf03cae1072922b554b3c98fe2b65a9322e0  base-04.png
90ed7f792f79dd4feb04e313b454e6632ccc3d42ab92363dc5f1f917e0d15816  base-05.png
19ce0121103f843e9908b0145444efc8de6dca46cc043e43e1da6813d842620e  base-06.png
fb92e005eb6e96375681a5af643d6a85c3074457331cae08b208774651a64f65  base-07.png
9c34c1c9e4cf8f2c0f0c3b2a64a44bf9dcfc0ab17fe30742621e9c263f6b96d4  base-08.png
2131d717d27c128b8f401343fa01fdd47d4b3680734b6bb57d3b77799594a570  lora-01.png
c73959e91eb8ac21b589ed4cbe271983af28774aaaf51aca4cd2d42260b5236d  lora-02.png
324ae7d5848d9b8725bbed9aa59d741eb1c2a4829774dc43a8c63d301d0acece  lora-03.png
a8bad78a682b31db45fc4687e90a26dcc24f28e548df5f1d51b38d2cc2c58a8f  lora-04.png
219c418f52bd315647ef3dd96907c25d935ae8e8fbee4e689f8af6d9b28757fe  lora-05.png
c2210ba08842656204aad99f024c2a9c2628337ef068f644e0cfd117376bc025  lora-06.png
fddf53430a90708949242727c9bc1353772d095ee997cf86b2736a4a5aa9eb65  lora-07.png
68041f2e5afe7994e85a65d0a020652ca27dcd58daa7188a532b430e08dd6449  lora-08.png
```

La GPU quedó a `0 %` de utilización y `0 MiB` de memoria ocupada al cerrar la
sesión. La instancia se mantiene encendida a la espera de la decisión del
usuario (apagar o continuar con escala-2).
