# ADR 0008: Destilación progresiva del prior a 256 px como vía de integración móvil

- Estado: aceptado.
- Fecha: 2026-08-17.

## Contexto

El ADR 0007 registró la brecha de órdenes de magnitud entre la receta oficial
de Würstchen v2 (30 timesteps, 1024 px, prior de ~994 M parámetros en fp16) y
el presupuesto móvil (RNF-01 a RNF-03), y bloqueó cualquier benchmark del
modelo real en el dispositivo hasta que exista un artefacto destilado, dejando
la elección del mecanismo a este ADR.

El ciclo de escalado LoRA quedó cerrado con escala-4
(`docs/experiments/wuerstchen-lora-scale-4-2026-08-17.md`): ni más pasos, ni
más datos, ni 512 px nativos, ni un LoRA ampliado superaron la fidelidad
parcial actual. El mejor checkpoint *limpio* es el de escala-3 (500 pasos,
`lr=5e-5`, 256 px, rango 16, sólo atención; SHA-256 `fbc61da9...`); el de
escala-4 añade artefactos por memorización de la franja de firma y no mejora
la fidelidad, por lo que queda descartado como maestro.

## Decisión

1. **Maestro de destilación**: prior base + LoRA de escala-3, ejecutado con la
   receta oficial (30 timesteps, guía 8.0), a **256 × 256**, resolución de
   producto fijada por ADR 0007.
2. **Mecanismo**: destilación progresiva del prior (reducción iterativa de
   pasos a la mitad, estudiante inicializado desde el maestro con la misma
   arquitectura), objetivo de la primera iteración: **30 → 8 pasos** en dos
   etapas (30→16, 16→8). La guía classifier-free se destila dentro del
   estudiante para no duplicar el coste por paso en el dispositivo.
3. **Línea base sin entrenamiento primero**: antes de pagar la destilación, se
   evalúa visualmente el prior de escala-3 con el scheduler actual reducido a
   12 y a 8 timesteps. Esa medición es el piso que la destilación debe superar;
   si la calidad a 8 pasos sin destilar ya es aceptable, la destilación se
   simplifica o se descarta.
4. **Decoder y VQGAN** se mantienen sin cambios en la primera iteración (12
   pasos); su reducción se evalúa después, con evidencia.
5. **Cuantización**: el artefacto destilado se exporta a ONNX y se cuantiza
   con la política selectiva del ADR 0006 antes de benchmarkar.
6. **Tamaño**: si el estudiante (994 M parámetros, ~1 GB en INT8) supera el
   presupuesto RNF-01 (≤ 250 MB), la reducción arquitectónica del estudiante
   requerirá un ADR propio; no se asume en este.

## Presupuesto de validación

- **Visual**: los ocho prompts fijos del conjunto de evaluación congelado
  (`docs/lora-scale-1-design.md`), con la lista de verificación vigente
  (rostro válido, atributos, edad aparente adulta — RF-09, divergencia vs.
  base-only). Cada iteración de destilación se compara contra el maestro a 30
  pasos y contra la línea base sin destilar al mismo número de pasos.
- **Dispositivo**: benchmark en el TECNO KM5s (serial obligatorio) sólo del
  artefacto exportado y cuantizado, contra RNF-01, RNF-02 y RNF-03; las
  métricas de emulador no cuentan (regla vigente).
- **Coste**: cada iteración de destilación cabe en una sesión de GPU alquilada
  (RTX 4090); los pesos y datasets se restauran con
  `scripts/bootstrap-vast.sh` y todo artefacto vuelve a la máquina local con
  verificación SHA-256, como en las escalas 1-4.

## Consecuencias

- Se desbloquea la integración del modelo real: el pipeline es destilación
  progresiva → exportación ONNX → cuantización selectiva → benchmark en
  dispositivo físico.
- El checkpoint de escala-4 queda como evidencia, no como maestro; cualquier
  nueva corrida LoRA queda descartada salvo decisión explícita del usuario.
- La línea base sin entrenamiento (punto 3) puede responder pronto y barato si
  hace falta destilar siquiera: es la primera tarea ejecutable de la Fase 3.
- La destilación requiere entrenamiento en GPU: se alquilará instancia bajo
  decisión del usuario, con el flujo de transferencia y verificación vigente.
