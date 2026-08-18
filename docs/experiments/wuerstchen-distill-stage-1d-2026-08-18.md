# Destilación del prior, etapa 1d (30→15 pasos, trayectorias reales) — 2026-08-18

Cuarta iteración de la etapa 1 y primera con la reformulación del ADR 0009
(`docs/distill-trajectory-design.md`): el estudiante se entrena sobre los
**latentes reales de las trayectorias guiadas del maestro** (una trayectoria
de 30 pasos por cada uno de los 818 captions de train, generada una sola vez;
el entrenamiento no vuelve a ejecutar el maestro). Única variable cambiada
respecto a la etapa 1c: la distribución de entrada. Resultado: **compuerta no
superada (0/8)**. Conforme al punto 4 del ADR 0009, no se pagan más
iteraciones del mecanismo de halving: la vía de destilación progresiva del
prior de 994 M queda **cerrada con evidencia**.

## Configuración ejecutada

| Parámetro | Valor |
|---|---|
| Maestro | prior base + LoRA escala-3 (SHA-256 `fbc61da9...`), guía 8.0, rejilla oficial de 30 |
| Trayectorias | 818 (una por caption de train v2.1.0), latentes de los 30 puntos en bf16 + embeddings de texto cacheados |
| Estudiante | prior completo (994 M) con LoRA fusionado, fp32, AdamW `lr=5e-5` |
| Rejilla del estudiante | 15 puntos (pares de la oficial), 14 saltos |
| Pérdida | MSE/‖eps‖² por muestra (`--normalize-target`) |
| Pasos | 6000, semilla 42, `--mode trajectory` |
| Checkpoint | `artifacts/distill-stage-1d/pilot-checkpoint.pt` (sólo en la instancia; hashes en `artifacts/distill-stage-1d/SHA256SUMS`, SHA-256 del checkpoint `3b455376...`) |

## Trayectoria de pérdida (escala normalizada, no comparable con 1/1b)

paso 1 = 0.412057, 500 = 0.030359, 1000 = 0.609951, 2000 = 0.445369,
3000 = 0.242699, 4000 = 0.667841, 5000 = 0.714163, 6000 = 0.250125. Misma
oscilación por salto muestreado que en 1c, sin pérdidas casi nulas.

## Evaluación visual (15 timesteps, 8 prompts fijos, 1024 px, `--skip-base`)

Las 8 muestras pasaron `validation_ok`. Evaluación visual:

| Muestra | Resultado |
|---|---|
| lora-01 | Masa de pelo negro sobre forma blanca sin rasgos → no válida |
| lora-02 | Campo de color plano → no válida |
| lora-03 | Rostro con **tres filas apiladas de gafas y ojos** → no válida (repetición) |
| lora-04 | Silueta de cabeza con textura de mosaico en el pelo y **dos bocas apiladas** → no válida |
| lora-05 | Bloques abstractos sin rostro → no válida |
| lora-06 | Cabeza con un solo ojo y pelo de puntos repetidos → no válida |
| lora-07 | Forma de pelo negro sin rasgos → no válida |
| lora-08 | **Múltiples rostros superpuestos** (≥3, con rasgos completos) → no válida |

**0/8 avatares únicos válidos; compuerta de etapa 2 (≥ 6/8) no superada.**

## Diagnóstico

1. La hipótesis del ADR 0009 queda **refutada como causa única**: entrenar
   sobre la distribución real de estados del maestro no eliminó el defecto de
   composición. El patrón dominante vuelve a ser la **repetición** (03, 04,
   08: rasgos completos pero duplicados/apilados), igual que en 1c.
2. La repetición sistemática apunta al **mecanismo de halving en sí**: un
   estudiante con la misma arquitectura que aprende saltos dobles como un
   epsilon efectivo único produce pasos que «sobre-actúan» de forma coherente;
   ni la señal (1b/1c) ni la distribución de entrada (1d) lo corrigen con este
   presupuesto de entrenamiento (6000 pasos, 818 captions, un caption por
   paso).
3. Cuatro formulaciones con evidencia (SNR, uniforme, normalizado,
   trayectorias reales) agotan la destilación progresiva del prior de 994 M a
   15/8 pasos dentro del presupuesto fijado. Conforme al ADR 0009, las
   opciones restantes son: **(c)** aceptar el modelo en GPU/servidor y
   replantear el alcance offline, o una **reducción arquitectónica profunda**
   del estudiante (modelo menor entrenado o destilado desde cero, ADR propio).

## Estado al cierre

PNG de evaluación, logs y manifiesto SHA-256 en `artifacts/distill-stage-1d/`
(local y en la instancia); el checkpoint de 2 GB permanece sólo en la
instancia (sin valor de continuidad tras fallar la compuerta). La GPU quedó al
0 % de utilización. La decisión siguiente es del usuario; no hay instancias
que mantener encendidas para esta vía.
