# Experimento LoRA escala-4 — 2026-08-17

Resultados del experimento diseñado en `docs/lora-scale-4-design.md`: una sola
corrida que prueba a la vez las dos palancas de capacidad restantes —
resolución de entrenamiento 512 px y LoRA ampliado (rango 32, atención + FFN)
— sobre la release v2.2.0 (1024 muestras nativas de 512 px).

## Configuración ejecutada

| Parámetro | Valor |
|---|---|
| Dataset | v2.2.0, `data/training-procedural-v2-2` (lock `dataset-2.2.0.lock.json`) |
| Pasos / lr / semilla | 500 / 5e-5 / 42 |
| Resolución de entrenamiento | 512 (latentes de 16×16; escala 1-3 usaban 8×8) |
| LoRA | rango 32, alpha 32, dropout 0.05 |
| Módulos LoRA | `to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2` |
| Parámetros entrenables | 12,582,912 (verificado en el arranque) |
| Instancia | RTX 4090 nueva, bootstrap automatizado con verificación completa |

Pérdidas de referencia: paso 1 = 0.000580, 100 = 0.043624, 200 = 0.094459,
300 = 0.023222, 400 = 0.001726, 500 = 0.038033 (pérdida por muestra única;
el rango coincide con escala 1-3). La validación visual usó la receta oficial
(30 timesteps, fp16, 1024 px, guía 8.0, prompt negativo con términos de
menores, seed 42) con los ocho prompts fijos; 16/16 muestras `validation_ok`
con embeddings finitos.

## Evaluación visual

| # | Atributos clave del prompt | LoRA escala-4 | Divergencia vs. base-only |
|---|---|---|---|
| 1 | happy, square, porcelain, side-parted black, green eyes, earrings, sky | adulto válido; black hair, sky ✓; green eyes/earrings ✗ (verificado a resolución nativa) | clara; más plano y geométrico |
| 2 | confident, heart, light, bob brown, gray eyes, freckles, lavender | adulto válido; bob brown (con mechón gris), lavender ✓; gray eyes/freckles ✗ (verificado a resolución nativa); pie limpio | clara |
| 3 | calm, oval, deep, curly pink, blue eyes, round glasses, sky | adulto válido; curly pink, sky ✓; gafas presentes pero cuadradas ✗ (escala-3 las logró redondas); blue eyes ✗; **artefacto: ojos pequeños dibujados dentro del cabello** | clara |
| 4 | smiling, round, brown skin, short blue, brown eyes, mint | adulto válido; brown skin, smiling, brown eyes, mint ✓; cabello no azul ✗ | clara |
| 5 | happy, heart, golden, bob blonde, green eyes, coral | adulto válido; happy, coral ✓; **iris ámbar visible** (primera vez que el color de iris diverge del oscuro por defecto, aunque no es el verde pedido); cabello tipo gorra verde ✗ (mismo fallo que escala 1-3; la base-only sí logra el rubio) | clara |
| 6 | calm, square, tan, curly auburn, gray eyes, earrings, sand | adulto válido; tan ✓; fondo casi blanco en vez de sand ~; gray eyes/earrings ✗; **artefacto: rectángulo gris en la frente, similar a la franja de firma del dataset** | clara |
| 7 | confident, oval, brown skin, short black, brown eyes, lavender | adulto válido; brown skin, short black hair, brown eyes, lavender ✓ | clara; muy fiel al prompt |
| 8 | smiling, round, porcelain, side-parted pink, blue eyes, freckles, mint | adulto válido; side-parted pink (laterales azules), smiling, mint ✓; blue eyes/freckles ✗ (verificado a resolución nativa) | clara |

### Lista de verificación

1. **Rostro válido**: 16/16 sin ruido, mosaico ni imagen en blanco. ✓
2. **Atributos del prompt**: fidelidad parcial, mismo patrón que escala 1-3:
   fondos, expresiones, tonos de piel y tipo de cabello casi siempre
   correctos; color de ojos según prompt, pecas y pendientes siguen fallando.
3. **Edad aparente adulta (RF-09)**: 16/16 representan adultos. ✓
4. **Divergencia respecto a base-only**: clara en los 8 pares.

## Comparación con escala-3 (512 px + LoRA ampliado vs. 256 px + rango 16)

- **Sin mejora material en atributos difíciles**: color de ojos según prompt,
  pecas y pendientes fallan en los mismos casos; el fallo del rubio del caso 5
  persiste idéntico por cuarta corrida consecutiva.
- **Señal parcial nueva**: el caso 5 muestra iris de color ámbar, la primera
  variación real de color de iris en todas las escalas — la resolución extra
  mueve algo, pero no hasta la fidelidad pedida.
- **Artefactos nuevos**: el caso 6 reproduce un rectángulo gris similar a la
  franja de firma del dataset y el caso 3 dibuja ojos dentro del cabello. La
  capacidad extra (rango 32 + FFN) está memorizando patrones espurios del
  dataset, señal de que la capacidad del adaptador ya no es el límite útil.
- Las pérdidas de referencia son del mismo rango que en escala 1-3.

## Conclusión y siguiente paso

La corrida de escala-4 es **visualmente válida** según la lista del diseño, y
su resultado es el que la compuerta preveía: ni la resolución 512 nativa ni el
LoRA ampliado compran fidelidad en color de ojos, pecas ni pendientes. Tras
agotar más pasos (escala-2), más y mejores datos (escala-3) y más resolución y
capacidad (escala-4), la evidencia de que **el techo está en el modelo base
para estos rasgos finos** queda sólida. Conforme al diseño, no se paga ninguna
corrida más de este tipo: se acepta la fidelidad actual como techo del piloto
y la siguiente fase es la integración con destilación o reducción de pasos
según ADR 0007, que requerirá su propio ADR antes de cualquier benchmark del
modelo real en el dispositivo.

## Hashes de las muestras (SHA-256)

```text
b069f9bbe5651088ddeff71f5be4b496e4840d9985d96560c4faa1c4c7cbc4e2  pilot-checkpoint.pt
b5f1bbb02a3bc1c320b7b4ec58b4ded62f1258103d073f267c87bd251c35655f  eval/base-01.png
28c4a483c74feaa74a404c17580d689ce89174188461b54d70619afcf0c9b070  eval/base-02.png
c1bdd6dc639e87758dff8b779d5efbc93bdfcf1d612a412b4aaed04cde0c0cbe  eval/base-03.png
ca4c501209d4611574ceaa0278291ae7f24b4977beacd302c45670a9cfd69c44  eval/base-04.png
e7a188ab3fd28d1bcf09547b3069523b756290a8a309d9951184fc8300876714  eval/base-05.png
f90b06050318fec3478d1e9199513b33fd64efbcc06aedea35c33ab2fa3913ee  eval/base-06.png
de0ee3ee633b61b73a29b8a7b9626851eeeef5ab3a0dbc23b47ccece986211f1  eval/base-07.png
63ccb1d2d5b00d9d0bacfe4fb1f6be02da8ebeca618579a9142ec9c506a449f1  eval/base-08.png
be33ee7f4c297a5ec6a927f9b962522f4993c08639f39c55ee09a1f7bca9ee4f  eval/lora-01.png
602dcbf73c8cdd03f900b1c751b42f8776dc3ed57eff647b69fe65aa14eec51c  eval/lora-02.png
44cae0d9865734458280096dc4a196d605a23bd24cdb17533ae176c00a39f8cd  eval/lora-03.png
b9b97791c079bd4b6e659b7ab81ec3d2d2cbf1006c6c4787f9c4dd8a0ad3524a  eval/lora-04.png
6b817470c3497416dc25b6d8f8d31fa2195262d4bfc9cd63f50e08e20dc6b7ee  eval/lora-05.png
017b13db415bfaefb9f15ea02d5c391177d1aa68c690b561681954831fbb60dd  eval/lora-06.png
54e5056e134f9d7b4d0b93fe7f0e6fe6852514552f599ed5b473767bef4b92bb  eval/lora-07.png
aea7b349ff8b9bda60bdd1e2982505b83b0c024679592a1b2435fdf6978edcd9  eval/lora-08.png
a5b4756d7f2e7722311c17a8ceb6c3676849234aad1414bb4935e6650e6fd9f1  train-scale-4.log
0fb5ac5a5c0b36a8b312d87208c3dbc9e0d2216a6bef0a1100e738b86864c827  eval-scale-4.log
```

Los 18 archivos se transfirieron a la máquina local y verificaron contra
`artifacts/lora-scale-4/SHA256SUMS` (19/19 OK, incluido el propio checkpoint).
La GPU quedó a `0 %` de utilización y 1 MiB de memoria ocupada al cerrar la
sesión; la instancia queda encendida a la espera de que el usuario la
destruya desde la consola de Vast.ai.
