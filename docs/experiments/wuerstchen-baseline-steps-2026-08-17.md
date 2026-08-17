# Línea base de reducción de pasos sin entrenamiento (ADR 0008) — 2026-08-17

Resultados de la línea base prevista en el ADR 0008, punto 3: medir qué
calidad produce el checkpoint de escala-3 (maestro de destilación elegido)
simplemente reduciendo los timesteps del prior, sin entrenar nada. Es el piso
que cualquier destilación debe superar.

## Configuración ejecutada

| Parámetro | Valor |
|---|---|
| Checkpoint | escala-3 (500 pasos, `lr=5e-5`, 256 px, rango 16; SHA-256 `fbc61da9...`) |
| Timesteps del prior | 12 y 8 (subconjuntos uniformes de la rejilla oficial de 30, `--prior-timesteps`) |
| Resto de la receta | oficial: 1024 px, fp16, guía 8.0, 12 pasos de decoder, seed 42, prompt negativo con términos de menores |
| Muestras | los 8 prompts fijos del conjunto congelado, sólo LoRA (`--skip-base`); los base-only de escala-3 sirven de referencia |
| Instancia | RTX 4090 nueva, bootstrap automatizado; GPU al 0 % al cerrar |

16/16 muestras pasaron `validation_ok` (embeddings finitos, pixel_std dentro
de rango): la tubería no falla; lo que se degrada es el contenido.

## Evaluación visual

Criterio: «válida» = rostro con rasgos completos (ojos, nariz, boca);
«parcial» = le faltan rasgos; «inválida» = cara en blanco.

| # | 30 pasos (escala-3, referencia) | 12 timesteps | 8 timesteps |
|---|---|---|---|
| 1 | válida | inválida (sin rasgos) | inválida (en blanco) |
| 2 | válida (artefacto de texto ausente) | inválida (sólo hendiduras de ojos) | inválida (en blanco) |
| 3 | válida | **válida** (gafas, pecas visibles) | inválida (en blanco) |
| 4 | válida | parcial (ojos y boca, sin nariz) | inválida (en blanco) |
| 5 | válida (fallo del rubio) | **válida** (cabello rubio-anaranjado ✓) | inválida (sólo hendiduras) |
| 6 | válida | parcial (ojos y cejas, sin nariz ni boca) | inválida (mancha sin rasgos) |
| 7 | válida, muy fiel | inválida (sin rasgos) | inválida (en blanco) |
| 8 | válida | parcial (ojos cerrados y sonrisa) | inválida (en blanco) |

Totales: **12 timesteps → 2/8 válidas, 3 parciales, 3 inválidas; 8 timesteps
→ 0/8 válidas**. Todas las muestras con rostro representan adultos (RF-09);
las inválidas directamente no contienen rasgos.

## Conclusión y siguiente paso

La reducción ingenua de timesteps **no es viable**: a 12 pasos se pierden los
rasgos faciales en 6 de 8 muestras y a 8 pasos en todas. La línea base sin
entrenamiento queda muy por debajo de lo aceptable, así que la destilación
progresiva del prior (ADR 0008, punto 2) es **obligatoria**, no opcional: el
estudiante a 8 pasos debe aprender lo que el scheduler reducido no conserva.

Detalle alentador: las dos muestras válidas a 12 pasos conservan el estilo del
dataset (y el caso 5 muestra por primera vez cabello rubio-anaranjado con el
LoRA), lo que indica que la señal del maestro sí llega a los embeddings cuando
la trayectoria de muestreo no se degrada.

Siguiente tarea: diseñar la destilación progresiva 30→16→8 del prior a 256 px
(maestro = escala-3, guía destilada), con su documento de diseño previo a
pagar la corrida, siguiendo el patrón de `docs/lora-scale-1-design.md`.

## Hashes de las muestras (SHA-256)

```text
a3e64449e268f0f1b7ee742c643cf8698e647eed10c79e71cb003436b620d3eb  baseline-t12/lora-01.png
08e4ba741ae6daa0c76b7e5adfa86f20342329bb07026fe1dffea92c45b35a2b  baseline-t12/lora-02.png
14c056a24ffebbfb511ad6edf328f564574b3ca1dd3ed2a72bfb058c837d1e2b  baseline-t12/lora-03.png
1950122e1d450ce041c9344f5cf7f552f0c8b53b175550c4445f2fd2699122f6  baseline-t12/lora-04.png
625f42a159e414fd56ab586b5c8a3e161baf2ea7c80a810d983bd28f12be2b00  baseline-t12/lora-05.png
eb27e037b3291ad7f7cbaa15a44ce7a72556f805616ffa32b4fb8757a82f6587  baseline-t12/lora-06.png
cb006ab7739611265c4461a28bebfe7f3964bc199fbbc08d368b4f4411fffd20  baseline-t12/lora-07.png
12002c9626ab6b9414a04fb6bb1a10ea35df5ad81db88798ded260665f5bf9aa  baseline-t12/lora-08.png
872ccbceb3786e1fc6a2119170c4c739563a52a35c8712de1e6bc88828cbf812  baseline-t8/lora-01.png
10ece8a66380d6525ecefe760eb0aeb8fa18055be23730489e05dd3d8cb1d37e  baseline-t8/lora-02.png
44f50ef2c40eb3238e4615d520ed6e863844e70bd77de3439efa26d82289c6fa  baseline-t8/lora-03.png
e0d51f05c699cf26d399e5e82a146fd0d53c5b9b5bdd0829aca6484097c6b158  baseline-t8/lora-04.png
577e8b5b43ce60fdfbbd255d06db3048d40cd6980a6770c3a81d7767e33a02cf  baseline-t8/lora-05.png
53b93287130d481d6d2172d12b4e97367c6d1129e577032f1e34a3e309642190  baseline-t8/lora-06.png
8a31e032e1bf54191794d1fde3bfe0a8987f2d52e8718acfec4ce807a245aaa0  baseline-t8/lora-07.png
846b396d9d4a2db562244197e79d8b54625396768997ebed1188ca6bc67747ae  baseline-t8/lora-08.png
04839325a0e36ec6752c33c131079abc2dead224b3d24c6237266a452424a5b0  eval-baseline-t12.log
a40fe4f7c4fe215044cd8475b3a01ff315960531fee0d9fe550d5e0ac1a77f27  eval-baseline-t8.log
```

Los 18 archivos se transfirieron a la máquina local y verificaron contra
`artifacts/baseline-steps/BASELINE-SHA256SUMS` (18/18 OK). La GPU quedó a
`0 %` de utilización y 1 MiB de memoria ocupada al cerrar la sesión; la
instancia queda encendida a la espera de que el usuario la destruya desde la
consola de Vast.ai.
