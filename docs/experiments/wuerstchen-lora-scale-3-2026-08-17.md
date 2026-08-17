# Experimento LoRA escala-3 — 2026-08-17

Tercera corrida de escalado, autorizada por la compuerta de escala-2
(`docs/experiments/wuerstchen-lora-scale-2-2026-08-16.md`): 500 pasos con
`lr=5e-5`, misma configuración que escala-2, cambiando únicamente el dataset a
la release **v2.1.0** (818 muestras de train, generador v3 con captions
«of an adult», detalle de forma de ojos y accesorios ampliados; ver
`docs/phase-2-dataset-status.md`). Instancia Vast nueva (RTX 4090, 48 GiB;
IP y puerto registrados fuera de Git).

## Re-entrada automatizada

Segunda ejecución real del flujo completo (`scripts/bootstrap-vast.sh`), ya con
la copia de `SHA256SUMS` incorporada:

1. Los dos paquetes v2.1 (`avatarface-training-procedural-v2-1.tar`,
   `avatarface-smoke-procedural.tar`) y `SHA256SUMS` se transfirieron por ruta
   directa (`scp` a `/tmp/avatarface-transfer/`), verificados antes en local y
   después en la instancia (`sha256sum -c`: OK).
2. El bootstrap clonó el repo, instaló las dependencias fijadas, restauró ambos
   datasets con verificación SHA-256 y descargó los 33 archivos de pesos desde
   HuggingFace contra el manifiesto recortado
   (`model_manifest_ok files=33 bytes=24151097654`).
3. Lock del dataset v2.1.0 verificado, auditoría sin hallazgos (1024 muestras)
   y `preflight-vast` con `ready:true` (CUDA real, sin entrenar).

## Entrenamiento

- Script: `scripts/run_wuerstchen_lora_pilot.py --steps 500 --learning-rate
  5e-5 --output artifacts/lora-scale-3` (el parámetro `--output` es un
  directorio; el checkpoint queda en `pilot-checkpoint.pt`).
- LoRA: rango 16, alpha 16, dropout 0.05; sólo `to_q`, `to_k`, `to_v`,
  `to_out.0`. 6,291,456 parámetros entrenables (idéntico a escala-1/2).
- Semilla 42, resolución 256, batch 1, dtype bf16; 818 muestras distintas de
  train, sin repetición (a diferencia de escala-2, que agotaba las 408 de
  v2.0.0 en el paso 409 y empezaba un segundo ciclo).
- Duración: ~5 minutos de reloj en la RTX 4090.
- Pérdidas de referencia (paso: pérdida): 1: 0.000943, 51: 0.283498,
  101: 0.007252, 151: 0.045643, 201: 0.437412, 251: 0.209666,
  301: 0.386458, 351: 0.045838, 401: 0.579403, 451: 0.770605,
  500: 0.072818. Mismo patrón estable que en escala-2 (picos ocasionales
  < 1.0); sin inestabilidad con `lr=5e-5`.
- Checkpoint: `artifacts/lora-scale-3/pilot-checkpoint.pt` (12,675,061 B),
  SHA-256 `fbc61da942288997a59fecbc5fdf1ec2cdd88fcfbc4be88dee6617d45b013b47`.

## Evaluación visual (mismo conjunto congelado: 1024 px, 30 timesteps, guía 8.0, seed 42)

Los 8 prompts fijados del diseño se generaron con el checkpoint LoRA y con
`--base-only` (`scripts/eval-visual-lora.sh`). Las 16 salidas pasaron la
comprobación de degeneración (`pixel_std` entre 56.1 y 108.9) y la inspección
visual. Muestras en `artifacts/lora-scale-3/eval/` (SHA-256 abajo).

**Control de reproducibilidad:** a diferencia de escala-2 (cuyas 8 imágenes
base-only fueron bit-idénticas a las de escala-1), en esta instancia las
base-only **no** son bit-idénticas (hashes distintos a los de escala-1/2). La
imagen de la instancia cambió (se observa `torch 2.12.0+cu126` en el
preflight), así que el determinismo bit a bit no se sostiene entre imágenes de
instancia distintas. La comparación visual sigue siendo válida: la receta, la
seed y los prompts son los mismos.

| # | Atributos clave del prompt | LoRA escala-3 | Divergencia vs. base-only |
|---|---|---|---|
| 1 | happy, square, porcelain, side-parted black, green eyes, earrings, sky | adulto válido; happy, black hair, sky ✓; green eyes/earrings ✗ (verificado a resolución nativa) | clara; estilo plano y geométrico |
| 2 | confident, heart, light, bob brown, gray eyes, freckles, lavender | adulto válido; bob brown, lavender ✓; gray eyes/freckles ✗; **el artefacto de texto de escala-2 ya no aparece** (pie limpio, verificado a resolución nativa) | clara |
| 3 | calm, oval, deep, curly pink, blue eyes, round glasses, sky | adulto válido; **curly pink ✓, round glasses ✓**, deep skin, sky ✓; blue eyes ✗ | clara; más fiel que la base (gafas de ojo de gato) |
| 4 | smiling, round, brown skin, short blue, brown eyes, mint | adulto válido; **brown skin ✓**, smiling, mint ✓; cabello no azul ✗ | clara |
| 5 | happy, heart, golden, bob blonde, green eyes, coral | adulto válido; happy, coral ✓; cabello tipo gorra verde ✗ (mismo fallo que escala-1 y escala-2; la base-only sí logra el rubio) | clara |
| 6 | calm, square, tan, curly auburn, gray eyes, earrings, sand | adulto válido; tan, calm ✓; fondo casi blanco en vez de sand ~; earrings/gray eyes ✗ | clara |
| 7 | confident, oval, brown skin, short black, brown eyes, lavender | adulto válido; **brown skin, short black hair, brown eyes, lavender ✓** | clara; muy fiel al prompt |
| 8 | smiling, round, porcelain, side-parted pink, blue eyes, freckles, mint | adulto válido; pink hair con capa azul, smiling, mint ✓; blue eyes/freckles ✗ (verificado a resolución nativa: iris oscuros) | clara |

### Lista de verificación

1. **Rostro válido**: 16/16 sin ruido, mosaico ni imagen en blanco. ✓
2. **Atributos del prompt**: fidelidad parcial; fondos, expresiones y tonos
   de piel casi siempre correctos; color de ojos, pecas y pendientes débiles
   (igual que en escala-1/2). Gafas redondas ✓ en el caso 3.
3. **Edad aparente adulta (RF-09)**: 16/16 representan adultos; ninguna
   muestra dudosa. ✓
4. **Divergencia respecto a base-only**: clara en los 8 pares; el estilo se
   vuelve más plano y geométrico, en la dirección del dataset procedimental.

## Comparación con escala-2 (v2.0.0 → v2.1.0, mismos 500 pasos y `lr=5e-5`)

- **Mejora puntual**: el artefacto de texto ilegible tipo marca de agua del
  caso 2 en escala-2 ya no aparece en escala-3 (pie de la imagen limpio).
- **Sin cambio material en fidelidad de atributos difíciles**: color de ojos,
  pecas y pendientes siguen fallando en los mismos casos; el fallo del rubio
  del caso 5 persiste idéntico. El detalle nuevo de los captions v2.1
  (`eye_shape`, más accesorios) y el doble de muestras **no destrabaron** esos
  atributos.
- Las pérdidas de referencia son casi idénticas paso a paso a las de escala-2,
  pese al dataset distinto; el entrenamiento es igual de estable.

## Conclusión y siguiente paso

La corrida de escala-3 es **visualmente válida** según la lista del diseño.
Hallazgo principal: ni más pasos (escala-2) ni un dataset con el doble de
muestras y captions más finos (escala-3) compran fidelidad en color de ojos,
pecas ni accesorios pequeños. El cuello de botella ya no parece estar en el
volumen ni en el detalle textual del dataset procedimental, sino en la
capacidad del ajuste LoRA actual (rango 16, sólo atención, resolución de
entrenamiento 256) o en el propio techo del modelo base para estos rasgos
finos. Se descarta otra corrida de 500 pasos con cambios sólo de dataset; las
opciones naturales a decidir con el usuario son: subir la resolución de
entrenamiento, ampliar los módulos/rango del LoRA, o aceptar la fidelidad
actual como techo del piloto y pasar a la fase de integración (destilación,
ADR 0007).

## Hashes de las muestras (SHA-256)

```text
fbc61da942288997a59fecbc5fdf1ec2cdd88fcfbc4be88dee6617d45b013b47  pilot-checkpoint.pt
153e6db937799eee86ccc818ba12644ecc07792a831bee77356ab974a34a22c0  eval/base-01.png
65d1acb6b0e2453ac777d0530615134c799390f041e6e550ac62f8ecdeb8defd  eval/base-02.png
78fd97023345640542040c30f81e0503e247face585b13e09f48bc4b9ffc5173  eval/base-03.png
c7e70beae76d9d8442be37ecf306c9e512984f002ecc4d97f2da3a361b0268ca  eval/base-04.png
522cd87470255c6b25ccb9a7bceb5ec4b8ba2ab0c2978c4aaf3c1eb9cbc9861e  eval/base-05.png
a1f43500401fa6ceeaadd9f1408267bc94dbdfd91951970c4c93821203cca583  eval/base-06.png
35b2ebbbb0a77ef455605ee1088a40bcfd37f3f8048b60e62660fddf761e6165  eval/base-07.png
c5eef08e01dc39af5da29f564f69d21ee848c93592b61c8ed848ef1c99a36213  eval/base-08.png
dad3ce550501a825b2bb70094b1f93dc5faed6601318a86beca9b1edbdec4885  eval/lora-01.png
55166c9f0c14276998a0639c971fe64ea9cc68e90969e88fdb2b79d797e303de  eval/lora-02.png
53b46dbee72116399077715fa23bddeebb4518226b9f6bb792812931c67b3b67  eval/lora-03.png
ed30fde2cf04225b3686aef97374154bd515dc2c9238782214216a663b52cf4d  eval/lora-04.png
cf2777f02737d91beb349bea22bd8edcd8745d546143f4d8bb578eeb11ca56a8  eval/lora-05.png
5dc75b279b9577560144023c5852f6a5a82c71f880e8dc17beea545475ee9f2f  eval/lora-06.png
b562531b19b235a097b61040ddbc7c1e992d5c028e5691a8e992571ccb79ab7f  eval/lora-07.png
16ef166a22c0099cb22a530f9cd1e4fffc5afbfb67796d2db1913d0daf60058a  eval/lora-08.png
a98c3b2b27c6c96f1ee66c9481ecdfce59e8ab9820db85fa9e00f5e5c787eb86  train-scale-3.log
9ffee5eec5ac27fae260f326768041cce0d20aa2b04d0851281f4d8329da5d64  eval-scale-3.log
```

Los 18 archivos se transfirieron a la máquina local y verificaron contra
`artifacts/lora-scale-3/SHA256SUMS` (19/19 OK, incluido el propio checkpoint).
La GPU quedó a `0 %` de utilización y 16 MiB de memoria ocupada al cerrar la
sesión; la instancia queda encendida a la espera de la decisión del usuario.
