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
| O-14 | Usar CPU como baseline del perfil micro | CPU: 19.7 ms; NNAPI: 153.5 ms | Medio | Bajo | P0 | Aplicado |
| O-15 | Reducir particiones NNAPI antes del perfil target | Bridge INT8 llegó a 111 particiones y luego timeout | Alto | Medio | P0 | Bloqueado para grafo actual |
| O-16 | Filtrar el APK a `arm64-v8a` | APK debug reducido de 78 a 21 MB | Medio | Bajo | P1 | Verificado |
| O-17 | Cuantizar antes de desplegar target | Bridge INT8 preprocesado: −67.7 % tamaño y −30.6 % latencia CPU | Alto | Alto | P0 | Verificado |
| O-18 | Optimizar ONNX antes de calibrar y cuantizar | Bridge CPU bajó de 115.5 a 80.2 ms | Alto | Medio | P0 | Verificado |
| O-19 | Añadir un perfil intermedio antes de `target` | Bridge validado con 8.24 M parámetros | Alto | Bajo | P0 | Verificado |
| O-20 | Validar modelo y backend al extraer cada resultado ADB | Se detectó una lectura stale en la primera corrida INT8 | Alto | Bajo | P0 | Aplicado |
| O-21 | Mantener vivas las sesiones ORT | Pipeline selectivo tarda 532 ms en cargar | Medio | Bajo | P0 | Verificado en benchmark |
| O-22 | Perfilar encoder, denoiser y decoder por separado | Denoiser 66 %, decoder original 33 % del tiempo aislado | Alto | Medio | P0 | Verificado |
| O-23 | Reducir canales y convoluciones tardías del decoder | Decoder fast FP32: 12.1 ms frente a 39.6 ms original | Alto | Alto | P0 | Verificado técnicamente |
| O-24 | Cachear el embedding del prompt | Pipeline ejecuta encoder una vez en 1.45 ms | Bajo | Bajo | P1 | Verificado en benchmark |
| O-25 | Calibrar denoiser con tensores representativos | Calibración sintética dio error medio 0.06397 | Alto | Medio | P0 | Pendiente |
| O-26 | Excluir checksum y posproceso del cronómetro de inferencia | Perfil ORT detectó ~175 ms ajenos a `session.run` | Alto | Bajo | P0 | Corregido |
| O-27 | Usar composición selectiva persistente | 67.5 ms P50, 78.1 ms P95 y 10.52 MB | Alto | Medio | P0 | Baseline provisional |

Cada propuesta nueva debe indicar la métrica que pretende mover. Una
optimización sólo se acepta como efectiva después de medirla.
