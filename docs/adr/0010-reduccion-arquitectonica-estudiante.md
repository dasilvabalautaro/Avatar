# ADR 0010: Reducción arquitectónica — estudiante compacto sobre salidas del maestro

- Estado: aceptado.
- Fecha: 2026-08-19.

## Contexto

La destilación progresiva del prior de 994 M quedó **cerrada con evidencia**:
cuatro formulaciones fallaron la compuerta de etapa 1 el 2026-08-18 (SNR,
uniforme, normalizada, trayectorias reales; ADR 0008 y 0009). Además, aun
superando la compuerta, el artefacto violaba el presupuesto móvil: el prior en
INT8 (~1 GB) excede RNF-01 (objetivo ≤250 MB), y el decoder Stage B y el
encoder de texto CLIP tampoco caben. La reducción de pasos nunca bastaba por
sí sola.

Decisión del usuario (2026-08-19): **mantener el alcance offline** (ADR 0002)
y seguir por la reducción arquitectónica profunda, con presupuesto austero.
La opción (c) del ADR 0009 (GPU/servidor) queda como último recurso si esta
vía falla sus compuertas.

Hechos que sostienen la viabilidad:

- El dominio es estrecho: rostros de avatar de adultos a 256 × 256 con el
  vocabulario cerrado de atributos del generador procedimental v3; no se
  necesita generación de dominio abierto.
- El baseline Android (modelo sintético de 8.2 M → 67.5 ms P50 en el TECNO
  KM5s) implica que un estudiante de 30–60 M en INT8 (≈30–60 MB) cabe con
  holgura en RNF-01/02/03 incluso con varios pasos de inferencia.
- El maestro validado (prior base + LoRA escala-3, receta oficial) es una
  fábrica de pares caption → imagen con licencias limpias y sin rostros
  reales.

## Decisión

1. **Estudiante compacto que reemplaza todo el stack** (prior + Stage B +
   VQGAN + encoder CLIP): U-Net condicional en espacio de píxeles a
   256 × 256, presupuesto de 30–60 M de parámetros, diseñado y entrenado
   desde cero en este repositorio (licencia propia Apache-2.0).
2. **Condicionamiento sin encoder de texto pesado**: los captions del dominio
   usan el vocabulario cerrado del generador v3; un parser de atributos en el
   dominio (extensión de la lógica de `AvatarPrompt`, que además sigue
   aplicando el rechazo de menores, RF-09) produce atributos estructurados
   que el estudiante consume vía embeddings aprendidos. En el APK no viaja
   ningún encoder de texto de terceros.
3. **Dataset de destilación**: el maestro genera pares (caption, seed) →
   imagen con la receta oficial (1024 px, 30 pasos, guía 8.0, prompt
   negativo) y las imágenes se reducen a 256 px (Lanczos). Objetivo: 4096
   pares sobre captions muestreados del vocabulario v3 con seeds fijas;
   release congelada con manifiesto, hashes SHA-256 y lock, como las v2.x.
   Se admite un corte intermedio de 2048 pares para la primera corrida si el
   costo lo exige.
4. **Dos formulaciones con orden fijo**:
   - **A (primera): destilación directa a pocos pasos** — el estudiante
     aprende (ruido z, atributos) → imagen final del maestro por regresión
     (L1; perceptual opcional sólo si la compuerta falla por borrosidad).
     Inferencia en 1–4 pasos.
   - **B (respaldo): difusor pequeño de 8 pasos** — el mismo U-Net entrenado
     como predictor de ruido sobre el mismo dataset, muestreo DDIM de 8
     pasos. Una sola iteración.
5. **Compuertas y presupuesto**: protocolo de evaluación idéntico a los
   experimentos previos — ≥ 6/8 rostros válidos, 8/8 de adultos, fidelidad
   de atributos comparable al maestro (±1 caso). Si A falla, se paga B una
   vez; si B también falla, la evidencia cierra la vía offline con modelo
   generativo y las opciones se reducen a la (c) del ADR 0009. Tope de gasto
   de toda la fase: **40 h de RTX 4090** (≈ 15–20 h dataset, resto
   entrenamiento y evaluación). No se paga GPU hasta que los scripts y sus
   pruebas estén listos en local.
6. **Integración móvil sin cambios de método**: artefacto aprobado →
   exportación ONNX → cuantización selectiva INT8 (ADR 0006, compuerta
   RNF-06 ≤ 5 %) → benchmark ADB en el dispositivo de referencia →
   verificación de RNF-01/02/03.

## Consecuencias

- Diseño detallado en `docs/student-distill-design.md` antes de pagar GPU.
- Scripts nuevos: muestreador de captions de destilación,
  `scripts/generate_distill_dataset.py` (corre en la instancia GPU) y
  `scripts/train_student.py`; el parser de atributos y el U-Net del
  estudiante viven en `src/avatar_face/` bajo Clean Architecture, con pruebas
  que no requieren torch.
- El techo de fidelidad del estudiante es el del maestro (color de ojos,
  pecas, pendientes); heredarlo es el resultado esperado y no invalida la
  vía.
- `scripts/distill_wuerstchen_prior.py` queda como histórico; no se pagan más
  corridas de destilación del prior.
