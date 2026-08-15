# Diseño del experimento LoRA escala-1 — 2026-08-15

Diseño del siguiente entrenamiento LoRA sobre Würstchen v2 Stage C, posterior al
piloto v3 (`docs/experiments/wuerstchen-lora-pilot-v3-2026-08-15.md`). Responde
a la tarea P1 del HANDOFF: fijar pasos, tasa de aprendizaje y tamaño de la
muestra de evaluación visual **antes** de pagar cualquier corrida.

## Punto de partida

- La integración completa (entrenamiento → checkpoint → validación con la
  receta oficial) está verificada de extremo a extremo con 20 pasos y
  `lr=1e-5`; la muestra es un rostro de adulto válido, muy próximo a la salida
  `base-only`.
- Evidencia histórica: con `lr=1e-4` la muestra de 20 pasos quedó en blanco;
  con `lr=1e-5` fue estable. El rango útil conocido es estrecho.
- Dataset: release congelada `training-procedural-v2` v2.0.0 (512 muestras;
  408 train, 52 validation, 52 test; lock SHA-256
  `79ecdd3f36301c4462372be35e93f66cee3e52f51d6992050728da8dc84334a2`).
- El script `scripts/run_wuerstchen_lora_pilot.py` ya acepta `--steps` y
  `--learning-rate`; usa hasta `max(steps, 20)` muestras de train. No requiere
  cambios para escala-1.

## Configuración de escala-1

| Parámetro | Valor | Justificación |
|---|---|---|
| Pasos | 200 | 10× el piloto; usa 200 muestras distintas de train (de 408) |
| Learning rate | 5e-5 | Punto medio entre el valor estable (1e-5) y el destructivo (1e-4); busca señal de adaptación visible en 200 pasos |
| Batch | 1 | Sin cambios de infraestructura |
| Semilla | 42 | Comparabilidad con el piloto y `base-only` |
| Resolución de entrenamiento | 256 | Igual que el piloto |
| LoRA | rango 16, alpha 16, dropout 0.05 en `to_q`, `to_k`, `to_v`, `to_out.0` | Configuración ya verificada |
| Precisión | bf16 | Igual que el piloto |
| Límite de coste | ≤ 1 hora de GPU | El piloto de 20 pasos duró minutos; 200 pasos ≈ 10× |

Si la muestra de escala-1 resulta inválida (ruido, mosaico, blanco), se repite
una única vez con `lr=1e-5` y 200 pasos. Si es válida pero indistinguible de
`base-only`, ese resultado también es información: se documenta y se decide
escala-2 con más pasos, no con más lr.

## Conjunto fijo de evaluación visual

Ocho prompts fijados de antemano, derivados de la taxonomía de captions del
dataset y todos **de adultos** (RF-09): cada uno pasa por `validate-prompt`
antes de la corrida. Seed 42 y receta oficial en todos: 30 timesteps, fp16,
1024 px, guía 8.0 y el prompt negativo que ya incluye términos de menores.

1. `flat vector avatar face of an adult, happy expression, square face, porcelain skin tone, side-parted black hair, green eyes with earrings, sky background`
2. `flat vector avatar face of an adult, confident expression, heart face, light skin tone, bob brown hair, gray eyes with freckles, lavender background`
3. `flat vector avatar face of an adult, calm expression, oval face, deep skin tone, curly pink hair, blue eyes with round glasses, sky background`
4. `flat vector avatar face of an adult, smiling expression, round face, brown skin tone, short blue hair, brown eyes without accessories, mint background`
5. `flat vector avatar face of an adult, happy expression, heart face, golden skin tone, bob blonde hair, green eyes without accessories, coral background`
6. `flat vector avatar face of an adult, calm expression, square face, tan skin tone, curly auburn hair, gray eyes with earrings, sand background`
7. `flat vector avatar face of an adult, confident expression, oval face, brown skin tone, short black hair, brown eyes without accessories, lavender background`
8. `flat vector avatar face of an adult, smiling expression, round face, porcelain skin tone, side-parted pink hair, blue eyes with freckles, mint background`

Cada prompt se evalúa también con el checkpoint base (`--base-only`) usando la
misma seed, para medir la divergencia introducida por el LoRA.

### Lista de verificación por muestra

1. Rostro válido: sin ruido, mosaico ni imagen en blanco.
2. Atributos del prompt presentes (expresión, forma, tono, cabello, ojos,
   accesorio, fondo).
3. **Edad aparente adulta** (RF-09): la salida no debe representar a una
   persona menor de edad; cualquier muestra dudosa invalida la corrida.
4. Divergencia respecto a `base-only` documentada (idéntica, sutil, clara).

## Compuertas y respaldo

- No se paga ninguna corrida más larga hasta que la muestra de escala-1 sea
  visualmente válida según la lista anterior.
- Respaldo local con SHA-256 de `pilot-checkpoint.pt` y de las 16 imágenes
  (8 LoRA + 8 base-only) en `artifacts/lora-scale-1/`; los hashes se registran
  en el documento de resultados.
- Al terminar: detener la instancia Vast y verificar 0 % de utilización.
- El documento de resultados se creará como
  `docs/experiments/wuerstchen-lora-scale-1-AAAA-MM-DD.md` con pérdidas por
  paso, hashes y la evaluación visual completa.

## Escalera posterior (sólo si escala-1 es válida)

- **escala-2**: 500–1000 pasos sobre las 408 muestras de train; evaluar si hace
  falta una release de dataset v2.1 con más muestras y la plantilla de captions
  «of an adult» (la v2.0.0 congelada no describe menores, pero no marca la edad
  de forma positiva).
- Toda decisión de integración móvil queda bloqueada por el ADR 0007 (brecha
  entre la receta oficial y el presupuesto RNF-03).
