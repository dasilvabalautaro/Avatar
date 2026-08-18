# Destilación del prior, etapa 1b (30→15 pasos, peso uniforme) — 2026-08-18

Segunda iteración de la etapa 1 de la destilación progresiva
(`docs/distill-prior-design.md`, ADR 0008), tras la corrección del mecanismo
de pérdida: **peso uniforme** en lugar del peso SNR que anulaba el gradiente
en saltos de alto ruido (véase
`docs/experiments/wuerstchen-distill-stage-1-2026-08-18.md`).
Resultado: **compuerta no superada**; la etapa 2 no se paga, conforme al
diseño.

## Configuración ejecutada

| Parámetro | Valor |
|---|---|
| Maestro | prior base + LoRA escala-3 (SHA-256 `fbc61da9...`), guía 8.0, rejilla oficial de 30 |
| Estudiante | prior completo (994 M) con LoRA fusionado, fp32, AdamW `lr=1e-5` |
| Rejilla del estudiante | 15 puntos (pares de la oficial), 14 saltos |
| Pérdida | MSE sobre epsilon con **peso uniforme** (sin SNR) |
| Pasos | 2000, captions de train v2.1.0, semilla 42 |
| Checkpoint | `artifacts/distill-stage-1b/pilot-checkpoint.pt`, SHA-256 `6c69da904e2d5f6eeaa264b8eb6d0b5f60984a6dad56e50fd2dd4d841ec3cd91` |

## Trayectoria de pérdida (referencia)

paso 1 = 1.287757, 500 = 0.043268, 1000 = 1.662426, 1500 = 1.322973,
2000 = 1.301729. Ya no hay pasos con pérdida casi nula (el peso uniforme
restauró la señal en todos los saltos), pero la pérdida no converge: oscila
entre ~0.04 y ~1.66 según el salto muestreado.

## Evaluación visual (15 timesteps, 8 prompts fijos)

Las 8 muestras pasaron `validation_ok` (embeddings finitos, pixel_std en
rango). Evaluación visual:

| Muestra | Resultado |
|---|---|
| lora-01 | Silueta de cabeza sin rasgos → no válida |
| lora-02 | Campo de color plano → no válida |
| lora-03 | **Rostro completo** (ojos, nariz, boca, gafas), adulto → válida |
| lora-04 | Cabeza con orejas sin rasgos → no válida |
| lora-05 | Campo de color plano → no válida |
| lora-06 | Campo de color plano → no válida |
| lora-07 | Campo de color plano → no válida |
| lora-08 | Silueta de cabeza sin rasgos → no válida |

**1/8 válidas; compuerta de etapa 2 (≥ 6/8) no superada.**

## Diagnóstico

1. La corrección del peso funcionó a nivel de señal: desaparecieron las
   pérdidas casi nulas y el estudiante recupera estructura global en 3/8
   muestras (siluetas de cabeza centradas) — la etapa 1 con SNR ni siquiera
   lograba eso. La muestra 03 demuestra que la tubería completa
   (destilación → 15 pasos → Stage B/A) puede producir un rostro.
2. El cuello de botella restante es la **capacidad de aprendizaje**: 2000
   pasos con `lr=1e-5` sobre 103 captions no bastan para que el estudiante
   aprenda los 14 saltos con la precisión que exigen los rasgos faciales.
   La oscilación de la pérdida (0.04–1.66) refleja magnitudes de epsilon muy
   distintas entre saltos que el peso uniforme no corrige.
3. Opciones documentadas para una eventual iteración futura (decisión del
   usuario, requieren GPU nueva):
   - más pasos de entrenamiento y/o mayor `lr` (p. ej. 5e-5);
   - normalización del objetivo por magnitud del epsilon objetivo de cada
     salto (equivalente a un peso `1/||eps||²`, no a SNR);
   - trayectorias de destilación consistentes (muestrear los saltos en orden
     de generación, no uniformemente).

## Estado al cierre

Conforme al diseño, **no se paga la etapa 2** y se detiene el entrenamiento
de destilación. Los 8 PNG de evaluación, el log y el manifiesto SHA-256
quedan en `artifacts/distill-stage-1b/` (local y en la instancia); el
checkpoint de 2 GB permanece sólo en la instancia (sin valor de
continuidad tras fallar la compuerta, no se transfiere).
