# Diseño: estudiante compacto destilado desde el maestro (ADR 0010)

Estado: vigente. Fecha: 2026-08-19. Decisión rectora: `docs/adr/0010-reduccion-arquitectonica-estudiante.md`.

## Objetivo

Producir el artefacto móvil real: un generador texto → rostro de avatar de
adulto a 256 × 256 que quepa en RNF-01/02/03, entrenado desde cero sobre
salidas del maestro (prior base + LoRA escala-3, receta oficial).

## Etapas y costos estimados

| Etapa | Dónde | Costo estimado |
|---|---|---|
| E1. Scripts, parser de atributos, pruebas | local, sin GPU | 0 USD |
| E2. Dataset de destilación (4096 pares) | Vast RTX 4090, ~15–20 h | ~8–15 USD |
| E3. Entrenamiento formulación A + compuerta | Vast RTX 4090, ~6–10 h | ~5–10 USD |
| E4. (sólo si A falla) formulación B, una vez | Vast RTX 4090, ~6–10 h | ~5–10 USD |
| E5. Export ONNX + INT8 + benchmark ADB | local + dispositivo | 0 USD |

Tope de fase: 40 h de GPU (ADR 0010). Ninguna etapa GPU se paga sin que la
anterior haya cerrado su compuerta.

## E1 — Trabajo local previo (bloqueante de todo gasto)

1. **Parser de atributos** en dominio: caption/prompt (vocabulario cerrado del
   generador v3: pelo, ojos —forma y color—, piel, pecas, accesorios, etc.) →
   dataclass `AvatarAttributes`. Reutiliza la validación de `AvatarPrompt`
   (rechazo de menores intacto). Pruebas puras sin torch.
2. **Muestreador de captions de destilación**: genera la lista determinista de
   4096 pares (caption, seed) con semilla 42, cubriendo el vocabulario de
   forma balanceada; se serializa a JSON con hash para viajar a la instancia.
3. **`scripts/generate_distill_dataset.py`**: corre en la instancia; carga
   maestro + LoRA escala-3, receta oficial (1024 px, 30 pasos, guía 8.0,
   prompt negativo, fp16), reduce a 256 px Lanczos, escribe manifiesto con
   SHA-256 por muestra y config completa; reanudable por índice.
4. **`scripts/train_student.py`**: entrena A o B (`--formulation`), config en
   el checkpoint, semilla 42, logging JSON, reanudable.
5. **U-Net del estudiante** en `infrastructure/` con pruebas de contrato
   (formas, número de parámetros) que se saltan sin torch.

## E2 — Dataset de destilación

- Maestro congelado: pesos por SHA-256 ya fijados + checkpoint LoRA escala-3.
- Salida: `data/distill-teacher-v1/` con `manifest.json`
  (`schema_version`, receta, hashes) → congelar con `freeze-dataset` →
  paquete `transfer/` verificado por hash (flujo estándar).
- Corte intermedio permitido: 2048 pares bastan para entrenar la primera
  corrida de A; se amplía a 4096 sólo si la compuerta lo pide.

## E3/E4 — Estudiante

- **Arquitectura**: U-Net 256 px, canales base 64 (mult. 1-2-3-4), atención en
  32² y 16², FiLM/cross-attention sobre embeddings de atributos aprendidos;
  objetivo 30–60 M de parámetros.
- **A (directa)**: entrada ruido z ~ N(0, I) + atributos → imagen del maestro
  para ese (caption, seed); pérdida L1. Inferencia 1 paso (hasta 4 si se
  añade refinamiento). Riesgo conocido: borrosidad por regresión; mitigación
  escalonada: pérdida perceptual sólo si la compuerta falla por ese motivo.
- **B (respaldo)**: mismo U-Net como predictor de ε, entrenamiento de difusión
  estándar sobre el dataset, muestreo DDIM 8 pasos.
- **Presupuesto de entrenamiento**: ~50k pasos, batch 32, lr 1e-4 coseno,
  fp16; checkpoint y muestras de control cada 5k pasos.
- **Compuerta** (idéntica al protocolo previo): 8 casos fijos de evaluación
  visual → ≥ 6/8 rostros válidos, 8/8 adultos, atributos comparables al
  maestro (±1 caso). Los 8 casos y seeds se congelan en el diseño del
  experimento antes de la corrida.

## E5 — Integración móvil

Pipeline ya validado: exportación ONNX → cuantización selectiva INT8
(ADR 0006, degradación ≤ 5 %, RNF-06) → benchmark
`avatar-face benchmark-android --serial 14254155BM000874` → verificación de
RNF-01 (≤250 MB), RNF-02 (≤1 GB), RNF-03 (≤5 s). Después: sustituir el modelo
sintético del APK instrumental, correr fixtures de regresión congelados y
estabilidad RNF-04 en el rango API 31–36.

## Registro

Cada corrida GPU produce su documento en `docs/experiments/` con config,
hashes, resultado de compuerta y decisión, como hasta ahora.
