# Experimento LoRA escala-2 — 2026-08-16

Segunda corrida de escalado, autorizada por la compuerta de escala-1
(`docs/experiments/wuerstchen-lora-scale-1-2026-08-15.md`): 500 pasos con
`lr=5e-5` sobre las 408 muestras de train de la release v2.0.0, en una
instancia Vast nueva (RTX 4090, 48 GiB; IP y puerto registrados fuera de Git).

## Re-entrada automatizada

Primera ejecución real del flujo de re-entrada completa
(`scripts/bootstrap-vast.sh`) sobre una instancia recién contratada:

1. Los dos paquetes de datasets y `SHA256SUMS` se subieron por ruta directa
   (`scp` a `/tmp/avatarface-transfer/`), verificados antes en local.
2. `bootstrap-vast.sh` clonó el repo, instaló las dependencias fijadas del
   piloto, restauró ambos datasets con verificación SHA-256 y descargó los 33
   archivos de pesos desde HuggingFace contra el manifiesto recortado
   (`model_manifest_ok ... files=33`); lock del dataset v2.0.0 verificado
   (`79ecdd3f...`) y auditoría sin hallazgos (512 muestras, 512 hashes
   únicos).
3. Fallo menor encontrado: `preflight-vast` exige `transfer/SHA256SUMS`
   dentro del repo y el bootstrap no lo copiaba desde el directorio de
   transferencia; se copió a mano y el preflight reportó `ready:true`. El
   script quedó corregido en este commit.

## Entrenamiento

- Script: `scripts/run_wuerstchen_lora_pilot.py --steps 500 --learning-rate
  5e-5`.
- LoRA: rango 16, alpha 16, dropout 0.05; sólo `to_q`, `to_k`, `to_v`,
  `to_out.0`. 6,291,456 parámetros entrenables (idéntico a escala-1).
- Semilla 42, resolución 256, batch 1, dtype bf16; 408 muestras distintas de
  train (con repetición en el segundo ciclo a partir del paso 409).
- Duración: ~5 minutos de reloj en la RTX 4090.
- Pérdidas de referencia (paso: pérdida): 1: 0.000916, 51: 0.298726,
  101: 0.006880, 151: 0.044419, 201: 0.457322, 251: 0.210260,
  301: 0.361571, 351: 0.043512, 401: 0.592142, 451: 0.743778,
  500: 0.059481. Picos ocasionales < 1.0, mismo patrón estable que en
  escala-1; sin inestabilidad con `lr=5e-5`.
- Checkpoint: `artifacts/lora-scale-2/pilot-checkpoint.pt` (12,675,061 B),
  SHA-256 `b27a363ed82eceb59825ed3b6e7be1b27e01a53f0fdc24792dd1c00a8e53915f`.

## Evaluación visual (mismo conjunto congelado: 1024 px, 30 timesteps, guía 8.0, seed 42)

Los 8 prompts fijados del diseño se generaron con el checkpoint LoRA y con
`--base-only`. Las 16 salidas pasaron la comprobación de degeneración
(`pixel_std` sano) y la inspección visual. Muestras en
`artifacts/lora-scale-2/eval/` (SHA-256 abajo).

**Control de reproducibilidad:** las 8 imágenes `base-only` son
bit-idénticas a las de escala-1 (mismos SHA-256), lo que confirma que la
receta y la seed son deterministas entre instancias.

| # | Atributos clave del prompt | LoRA escala-2 | Divergencia vs. base-only |
|---|---|---|---|
| 1 | happy, square, porcelain, side-parted black, green eyes, earrings, sky | adulto válido; happy, black hair, sky ✓; green eyes/earrings ✗ | clara; estilo más plano y geométrico que la base |
| 2 | confident, heart, light, bob brown, gray eyes, freckles, lavender | adulto válido; lavender ✓; bob brown/freckles ✗; **artefacto: texto ilegible tipo marca de agua al pie** | clara |
| 3 | calm, oval, deep, curly pink, blue eyes, round glasses, sky | adulto válido; **curly pink ✓, round glasses ✓**, sky ✓ | clara; más fiel al prompt que la base (gafas de ojo de gato) y que escala-1 |
| 4 | smiling, round, brown skin, short blue, brown eyes, mint | adulto válido; **brown skin ✓**, smiling, mint ✓; cabello no azul ✗ | clara; más fiel al tono de piel |
| 5 | happy, heart, golden, bob blonde, green eyes, coral | adulto válido; coral ✓; cabello verdoso tipo gorra ✗ (mismo fallo que escala-1) | clara |
| 6 | calm, square, tan, curly auburn, gray eyes, earrings, sand | adulto válido; tan ✓, calm ✓; fondo casi blanco y earrings ✗ | clara |
| 7 | confident, oval, brown skin, short black, brown eyes, lavender | adulto válido; **brown skin, short black hair, lavender ✓** | clara; muy fiel al prompt |
| 8 | smiling, round, porcelain, side-parted pink, blue eyes, freckles, mint | adulto válido; pink hair, smiling, mint ✓; blue eyes/freckles ✗ | clara |

### Lista de verificación

1. **Rostro válido**: 16/16 sin ruido, mosaico ni imagen en blanco. ✓
2. **Atributos del prompt**: fidelidad parcial; fondos, expresiones y tonos
   de piel casi siempre correctos; color de ojos, pecas y accesorios débiles
   (igual que en escala-1). Mejora puntual en el caso 3 (gafas redondas).
3. **Edad aparente adulta (RF-09)**: 16/16 representan adultos; ninguna
   muestra dudosa. ✓
4. **Divergencia respecto a base-only**: clara en los 8 pares; el estilo se
   vuelve más plano y geométrico, en la dirección del dataset procedimental.

## Comparación con escala-1 (200 → 500 pasos)

No son visualmente idénticas: la divergencia se acentúa (rasgos más simples
y geométricos, orejas más grandes, sonrisas más esquemáticas) sin degradar la
validez del rostro. La fidelidad de atributos difíciles (color de ojos,
pecas, pendientes, rubio en el caso 5) **no mejoró** con 2.5× más pasos, lo
que sugiere que el límite ya no está en los pasos sino en el dataset (512
muestras, captions sin marca positiva de edad ni detalle fino de ojos y
accesorios). Se descarta escalar a 1000 pasos con este dataset.

## Conclusión y siguiente paso

La corrida de escala-2 es **visualmente válida** según la lista del diseño.
Hallazgo principal: más pasos con el mismo dataset ya no compran fidelidad de
atributos. El siguiente paso natural es la release de dataset **v2.1**
(sección P3 del HANDOFF): más muestras y plantilla de captions «of an adult»
con mayor detalle de ojos y accesorios, antes de cualquier corrida más larga.
El artefacto de texto de `lora-02` queda registrado como riesgo conocido del
modelo base (aparece también, con otros textos, en corridas base-only
históricas) y se vigilará en evaluaciones futuras.

## Hashes de las muestras (SHA-256)

```text
b27a363ed82eceb59825ed3b6e7be1b27e01a53f0fdc24792dd1c00a8e53915f  pilot-checkpoint.pt
69e7395c2984562377f4a1103ee78b693b93ea0d48bcc3992c1d8b799e5f9df0  eval/base-01.png
3586c4efa473105c6a73f450f77224ce968430d207b829f26c8a27edc3e31936  eval/base-02.png
dca6598070a5efcd6984ea46deb27b12c2c1fd2121e3a118c4de6e329f17928d  eval/base-03.png
10a739d2059890851d042701d69edf03cae1072922b554b3c98fe2b65a9322e0  eval/base-04.png
90ed7f792f79dd4feb04e313b454e6632ccc3d42ab92363dc5f1f917e0d15816  eval/base-05.png
19ce0121103f843e9908b0145444efc8de6dca46cc043e43e1da6813d842620e  eval/base-06.png
fb92e005eb6e96375681a5af643d6a85c3074457331cae08b208774651a64f65  eval/base-07.png
9c34c1c9e4cf8f2c0f0c3b2a64a44bf9dcfc0ab17fe30742621e9c263f6b96d4  eval/base-08.png
87e49f176fa5791093b30ce457081962355835d9c50540c57cc9e44ab459c8e8  eval/lora-01.png
b8521ca5b7e96a4b1f6629613ef6021535c4ed25202277059d5c2d24e02d03d2  eval/lora-02.png
a86a8e99cf095e2539c35464904892f7daa41e3e865e41415add4bebad2d4fca  eval/lora-03.png
aa4148edb68137e0995e553e8708237700777527497b1d058af3360616072f2b  eval/lora-04.png
8ea19319abf4c15e5dbcd0d89182b53c1a5e48bbec5a49e927285bbbc78ed508  eval/lora-05.png
92572b7de11aaf5134dd7d9201eaa665dda6ee5fbdd2da09a5bb81414bd29a2c  eval/lora-06.png
c3602663a81a869ebdf2f97cf41c7cc2eaebeaa2747016a3d6eeeff9476f5eb2  eval/lora-07.png
adfbf7eaa66b31b88eb854c0323f870788cc7f75891b0e50257d26bb08008d18  eval/lora-08.png
264d3c72fb123429227ac76b89fd73c861ed6f4cbbe8ff853049ac0265c91164  train-scale-2.log
3e9b657a8856f37ff853f110333f362ca6c306f97702641c8bbc0b9724b6e0ce  eval-scale-2.log
```

Los 18 archivos se transfirieron a la máquina local y verificaron contra
`artifacts/lora-scale-2/SHA256SUMS`. La GPU quedó a `0 %` de utilización al
cerrar la sesión; la instancia queda encendida a la espera de la decisión del
usuario.
