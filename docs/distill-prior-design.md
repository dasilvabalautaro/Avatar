# Diseño de la destilación progresiva del prior — 2026-08-17

Diseño de la primera iteración de destilación del prior Würstchen Stage C,
según el ADR 0008. La línea base sin entrenamiento ya demostró que reducir
timesteps no es viable (12 pasos → 2/8 rostros válidos; 8 pasos → 0/8;
`docs/experiments/wuerstchen-baseline-steps-2026-08-17.md`), así que la
destilación es obligatoria. La configuración queda fijada **antes** de pagar
cualquier corrida.

## Punto de partida

- Maestro: prior base + LoRA de escala-3 (500 pasos, `lr=5e-5`, 256 px, rango
  16; SHA-256 `fbc61da9...`), receta oficial: 30 timesteps, guía 8.0, 1024 px
  en validación. El de escala-4 quedó descartado como maestro por sus
  artefactos (ADR 0008).
- El prior opera sobre latentes EffNet de 16 canales: a resolución de producto
  (256 px) son mapas de 8×8. La destilación se hace en el espacio del prior
  (texto → embedding de imagen); **no necesita imágenes, sólo captions**.
- Dataset de destilación: los 818 captions del split train de la release
  v2.1.0 (congelada, hash verificado). No se generan imágenes ni se decodifica
  nada durante el entrenamiento.
- Infraestructura reutilizable: carga de pesos, scheduler
  `DDPMWuerstchenScheduler`, text encoder y LoRA ya verificados en
  `scripts/run_wuerstchen_lora_pilot.py` y `scripts/validate_wuerstchen_lora.py`.

## Mecanismo (destilación progresiva, Salimans & Ho)

Cada etapa entrena un estudiante que hace en **un paso** lo que el maestro
hace en **dos**:

1. Se muestrea un caption, un ruido y un índice de timestep de la rejilla del
   maestro (etapa 1: la oficial de 30).
2. El maestro ejecuta dos pasos con guía classifier-free 8.0 (ya combinada:
   el estudiante aprende la salida **guiada**, así en el dispositivo no se
   duplica el coste por paso) y produce el objetivo.
3. El estudiante predice el salto de dos pasos en uno; pérdida MSE entre la
   predicción y el objetivo del maestro, con el peso SNR del scheduler.
4. El estudiante de cada etapa se inicializa desde el resultado de la etapa
   anterior (etapa 1: desde el maestro con el LoRA ya fusionado).

## Configuración fijada

| Parámetro | Valor | Justificación |
|---|---|---|
| Etapas | 2: 30→15 y 15→8 pasos | Mitades de la rejilla; el «30→16→8» del ADR 0008 era nominal |
| Resolución de destilación | 256 (latentes 8×8) | Resolución de producto (ADR 0007) |
| Captions | 818 de train v2.1.0, ciclados en orden determinista | Sin descarga ni generación de imágenes |
| Pasos de entrenamiento | 2000 por etapa | ~2.4 recorridos del corpus de captions por etapa |
| Learning rate | 1e-5, AdamW, bf16 | El rango estable conocido del prior (escala 1-4) |
| Semilla | 42 | Regla del proyecto |
| Estudiante | copia completa del prior (994 M) con LoRA escala-3 fusionado | Misma arquitectura; exportación ONNX ya conocida |
| VRAM estimada | ~14 GB (maestro bf16 + estudiante + AdamW fp32) | Cabe en la RTX 4090 de 48 GiB |
| Límite de coste | ≤ 4 horas de GPU en total | Cada paso cuesta 4 forwards del maestro + 1 del estudiante a 8×8 latentes |

Script nuevo: `scripts/distill_wuerstchen_prior.py`, con `--stage` (1|2),
`--student-steps`, `--teacher` (checkpoint de la etapa anterior) y salida
`pilot-checkpoint.pt` con `distill_config` embebida, siguiendo el patrón del
piloto LoRA. Se añadirá al validador la opción de cargar el prior completo
destilado (sin LoRA) con `--prior-timesteps` igual a los pasos del estudiante.

## Conjunto de evaluación y compuertas

- Los **ocho prompts fijos** del conjunto congelado
  (`docs/lora-scale-1-design.md`), con la lista de verificación vigente:
  rostro válido, atributos, edad aparente adulta (RF-09), divergencia vs.
  base-only. Se compara contra el maestro a 30 pasos (ya existente, escala-3)
  y contra la línea base al mismo número de pasos (12 y 8: ya existente).
- Métrica numérica adicional: MSE y similitud coseno entre los embeddings del
  estudiante y los del maestro sobre los 103 captions de validation de v2.1.0
  (registrada en el documento de resultados).
- **Compuerta de etapa 2**: la etapa 1 (15 pasos) debe producir ≥ 6/8 rostros
  válidos; si no, no se paga la etapa 2 y se documenta.
- **Compuerta de éxito de la iteración**: el estudiante a 8 pasos debe
  producir ≥ 6/8 rostros válidos y fidelidad comparable al maestro (misma
  tasa de atributos correctos ±1 caso). Entonces sigue exportación ONNX y
  cuantización selectiva (ADR 0006) y benchmark en el TECNO KM5s.
- Respaldo local con SHA-256 de checkpoints y muestras; documento de
  resultados `docs/experiments/wuerstchen-distill-stage-N-AAAA-MM-DD.md`;
  instancia destruida al terminar (GPU al 0 %).

## Riesgos conocidos

- El estudiante hereda el techo de fidelidad del maestro (color de ojos,
  pecas, pendientes): la destilación no lo mejora, sólo lo conserva; es el
  comportamiento esperado y no invalida la corrida.
- Si la etapa 1 ya degrada rasgos, el problema es del mecanismo (pesos SNR,
  rejilla) y se ajusta antes de gastar la etapa 2.
- El tamaño del artefacto (~1 GB en INT8) puede superar RNF-01 (250 MB): ese
  problema es del ADR futuro de reducción arquitectónica, no de esta
  iteración.
