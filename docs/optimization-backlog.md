# Backlog de optimización

| ID | Sugerencia | Evidencia actual | Impacto | Esfuerzo | Prioridad | Estado |
|---|---|---|---|---|---|---|
| O-01 | Validar exportación con un modelo mínimo antes del entrenamiento serio | La exportabilidad es un riesgo alto | Alto | Medio | P0 | Pendiente |
| O-02 | Presupuestar parámetros por encoder, denoiser y autoencoder | El encoder puede dominar el paquete | Alto | Bajo | P0 | Pendiente |
| O-03 | Usar 256 × 256 como perfil obligatorio inicial | Reduce cómputo y memoria frente a 512 | Alto | Bajo | P0 | Aceptado |
| O-04 | Congelar prompts y seeds de regresión desde el baseline | Permite atribuir cambios de calidad | Alto | Bajo | P0 | Pendiente |
| O-05 | Descargar y restaurar paquetes de uno en uno | Reduce el pico de disco en Vast.ai | Medio | Bajo | P1 | Aceptado |
| O-06 | Comparar INT8 antes de invertir en INT4 | INT4 puede no acelerar el backend real | Alto | Medio | P0 | Aceptado |
| O-07 | Medir cold y warm start por separado | La carga puede ocultar el costo de inferencia | Medio | Bajo | P1 | Pendiente |
| O-08 | Mantener el scaffolding sin frameworks hasta necesitarlos | Reduce acoplamiento y dependencias | Medio | Bajo | P1 | Aplicado |
| O-09 | Reducir el presupuesto máximo del proceso a 1 GiB | El TECNO KM5s mostró sólo 1.14 GiB disponible | Alto | Bajo | P0 | Aplicado |
| O-10 | Medir presión de memoria en series, no sólo una inferencia | El objetivo físico tiene 3.66 GiB de RAM | Alto | Medio | P0 | Pendiente |
| O-11 | Construir un modelo de viabilidad con pesos aleatorios | Detecta operadores inviables antes de entrenar | Alto | Medio | P0 | Aceptado |
| O-12 | No descargar AuraFlow/Kandinsky en el primer ciclo | Su tamaño no aporta evidencia móvil temprana | Medio | Bajo | P1 | Aceptado |
| O-13 | Mantener FLUX y Würstchen sólo como maestros potenciales | Ninguno cabe en el presupuesto Android | Alto | Bajo | P0 | Aceptado |

Cada propuesta nueva debe indicar la métrica que pretende mover. Una
optimización sólo se acepta como efectiva después de medirla.
