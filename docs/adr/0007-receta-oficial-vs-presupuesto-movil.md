# ADR 0007: Brecha entre la receta oficial de Würstchen y el presupuesto móvil

- Estado: aceptado.
- Fecha: 2026-08-15.

## Contexto

La receta oficial validada para Würstchen v2 (compuerta `base-only` y piloto
LoRA v3) genera rostros de avatar válidos con: 1024 × 1024 px, 30 timesteps en
Stage C, 12 pasos en el decoder, guía 8.0, fp16 y un prior de ~994 M
parámetros. Esa receta se ejecuta en una RTX 4090 de 48 GiB; nada de ella cabe
en el presupuesto del producto:

- RNF-03 exige ≤ 5 s a 256 × 256 en el TECNO KM5s (MT6769, 3.66 GiB RAM).
- RNF-01/RNF-02 fijan ≤ 250 MB de modelo y ≤ 1.0 GB de memoria; el prior en
  fp16 ya ocupa ~2 GB sólo en pesos.
- El baseline Android medido (67.5 ms P50) corresponde a modelos sintéticos de
  8.2 M parámetros; no es extrapolable al modelo real, que es ~120× mayor y
  además ejecuta decenas de pasos de difusión.

La brecha, por tanto, no es de porcentaje sino de órdenes de magnitud: ejecutar
la receta oficial tal cual en el dispositivo de referencia es inviable hoy.

## Decisión

- La receta oficial se reserva para entrenamiento y evaluación visual en GPU;
  no es candidata a ejecución en el dispositivo.
- Toda integración móvil del modelo real queda bloqueada hasta cerrar la brecha
  mediante destilación o reducción de pasos (p. ej. destilar el prior a pocos
  pasos a 256 px, resolución de producto) seguido de la cuantización selectiva
  del ADR 0006.
- 1024 px queda descartado como resolución de producto; el contrato móvil sigue
  siendo 256 × 256 (512 sólo si el dispositivo cumple los límites, según los
  requisitos de entrada).
- La calidad se entrena y valida con la receta oficial; el artefacto móvil
  deriva de ella por destilación. Ninguna métrica móvil del modelo real se
  acepta hasta que exista ese artefacto destilado.

## Consecuencias

- El escalado LoRA (ver `docs/lora-scale-1-design.md`) puede avanzar en GPU sin
  esperar decisiones móviles: produce el modelo de calidad que luego se
  destilará.
- Se necesita un ADR futuro que elija el mecanismo concreto de destilación o de
  reducción de pasos y su presupuesto de validación (muestreo visual +
  benchmark en dispositivo físico, RNF-09).
- El backlog de optimización (`docs/optimization-backlog.md`) y los perfiles
  sintéticos `target`/`stress` no miden esta brecha; sirven sólo para contratos
  y rendimiento de componentes.
