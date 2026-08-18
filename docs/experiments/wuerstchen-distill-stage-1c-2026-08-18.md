# Destilación del prior, etapa 1c (30→15 pasos, objetivo normalizado) — 2026-08-18

Tercera iteración de la etapa 1 (`docs/distill-prior-design.md`, ADR 0008),
tras los fallos de la etapa 1 (peso SNR) y la etapa 1b (peso uniforme).
Cambios: pérdida **normalizada por la potencia media del epsilon objetivo**
(`--normalize-target`, mse/‖eps‖²) y más capacidad de aprendizaje
(`lr=5e-5`, 6000 pasos). Resultado: **compuerta no superada**; conforme al
diseño y a lo acordado con el usuario, la vía de destilación queda descartada
salvo decisión contraria.

## Configuración ejecutada

| Parámetro | Valor |
|---|---|
| Maestro | prior base + LoRA escala-3 (SHA-256 `fbc61da9...`), guía 8.0, rejilla oficial de 30 |
| Estudiante | prior completo (994 M) con LoRA fusionado, fp32, AdamW `lr=5e-5` |
| Rejilla del estudiante | 15 puntos (pares de la oficial), 14 saltos |
| Pérdida | MSE/‖eps‖² por muestra (peso uniforme por salto) |
| Pasos | 6000, captions de train v2.1.0, semilla 42 |
| Checkpoint | `artifacts/distill-stage-1c/pilot-checkpoint.pt` (sólo en la instancia; hashes en `artifacts/distill-stage-1c/SHA256SUMS`) |

## Trayectoria de pérdida (escala normalizada, no comparable con 1/1b)

Los valores oscilan ~0.2-0.6 durante toda la corrida (pérdida final 0.2296);
la normalización elimina la dispersión de órdenes de magnitud entre saltos
que mostraban las etapas anteriores.

## Evaluación visual (15 timesteps, 8 prompts fijos)

Las 8 muestras pasaron `validation_ok`. Evaluación visual:

| Muestra | Resultado |
|---|---|
| lora-01 | 3 rostros superpuestos con rasgos → no válida (multi-rostro) |
| lora-02 | Mosaico de ~9 rostros completos → no válida |
| lora-03 | Grupo de ~8 rostros con gafas → no válida |
| lora-04 | 3 rostros (central con rasgos completos) → no válida |
| lora-05 | Mosaico de ~9 rostros rubios → no válida |
| lora-06 | Mosaico de ~9 rostros → no válida |
| lora-07 | Mosaico de ~10 rostros → no válida |
| lora-08 | 4 rostros deformes → no válida |

**0/8 avatares únicos válidos; compuerta de etapa 2 (≥ 6/8) no superada.**

## Diagnóstico

1. La combinación normalización + `lr=5e-5` resolvió el problema de señal:
   **los 8/8 generan rostros con rasgos completos** (1b lograba 1/8) y el
   condicionamiento por caption se nota (gafas en el caso 3, rubios en el 5).
2. El defecto restante es de **composición espacial**: el estudiante replica
   la cara en mosaico en lugar de componer una sola. Hipótesis: con el lr
   alto cada salto del estudiante «sobre-actúa» (aplica demasiada estructura
   por paso) y la repetición es la firma de ese exceso coherente en los 14
   saltos.
3. Las tres iteraciones agotan las correcciones baratas del objetivo
   (pesos, normalización, lr/pasos). Lo que queda — trayectorias
   consistentes, menor lr con muchos más pasos, o destilación sobre la
   trayectoria real del maestro en lugar de la marginal N(0, I) — exige
   reformular el método, no ajustar hiperparámetros.

## Estado al cierre

Conforme a lo acordado: **la vía de destilación progresiva queda descartada
con evidencia sólida** (tres formulaciones del objetivo, tres fallos de
compuerta). PNG de evaluación, logs y manifiesto SHA-256 en
`artifacts/distill-stage-1c/`; los checkpoints de 1b y 1c permanecen sólo en
la instancia hasta que el usuario la destruya. La integración móvil requiere
un replanteo (ADR nuevo): opciones registradas en el HANDOFF.
