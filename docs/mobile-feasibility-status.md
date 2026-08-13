# Estado del spike de viabilidad móvil

Este spike adelantó riesgos de las fases 5–7 antes de comenzar la Fase 2 formal
del plan maestro. No sustituye la fase de dataset legal y auditable.

## Resultado

El recorrido técnico PyTorch → ONNX → Android quedó demostrado en el teléfono
físico. El modelo utilizado tiene pesos aleatorios: esta fase valida arquitectura,
exportación y runtime, no calidad visual.

## Completado

- Perfiles sintéticos `micro`, `bridge`, `bridge-slim`, `bridge-fast`, `target` y
  `stress` con presupuesto de parámetros.
- Modelo de viabilidad desacoplado mediante puerto de exportación.
- Exportación ONNX opset 17, verificación estructural y comparación numérica con
  ONNX Runtime de escritorio.
- Proyecto Android Kotlin con ONNX Runtime 1.23.2.
- Benchmark CPU y NNAPI sobre TECNO KM5s, Android 15, por USB.
- Runner ADB reproducible con serial obligatorio y resultados JSON.
- APK limitado a `arm64-v8a`.
- Dependencias de viabilidad fijadas en `requirements-feasibility.lock`.

## Evidencia

- Modelo `micro`: 304,415 parámetros y 1,235,972 bytes en ONNX.
- Diferencia máxima PyTorch↔ONNX Runtime: `8.195638656616211e-08`.
- Mediana CPU corregida: 19.7 ms en siete corridas.
- Mediana con NNAPI solicitado corregida: 153.5 ms en siete corridas.
- PSS máximo muestreado: aproximadamente 118–119 MiB.
- APK debug: reducción aproximada de 78 MB a 21 MB al conservar sólo ARM64.
- Calidad del repositorio: 26 pruebas, Ruff limpio y mypy estricto limpio.

La evidencia detallada está en
`docs/experiments/android-micro-onnx-2026-08-13.md`.

## Decisión

CPU queda como baseline provisional del grafo `micro`. NNAPI participó, pero el
grafo se dividió en ocho particiones y el costo de coordinación superó el ahorro
de cómputo. La selección definitiva se hará con el grafo cuantizado candidato,
no con este microbenchmark.

## Siguiente compuerta

Implementar cuantización post-training INT8 sobre un perfil intermedio, comparar
tamaño, similitud numérica, latencia, memoria y particionamiento CPU/NNAPI, y sólo
después intentar el perfil `target`. Esto reduce el riesgo de llevar un grafo de
aproximadamente 136 millones de parámetros a Android sin evidencia previa.

La ruta QDQ ya fue probada sobre `micro`: redujo los pesos 34.9 %, pero CPU no
aceleró y NNAPI empeoró. El informe complementario está en
`docs/experiments/android-micro-int8-onnx-2026-08-13.md`.

El perfil intermedio `bridge` aportó una mejora material: INT8 con
preprocesamiento redujo tamaño 67.7 %, latencia CPU 30.6 % y PSS máximo muestreado
19.7 %. NNAPI fue descartado para este grafo tras alcanzar 111 particiones y
timeouts. Véase `docs/experiments/android-bridge-int8-onnx-2026-08-13.md`.

El perfilado aislado y la corrección del cronómetro mostraron que el denoiser
original consume aproximadamente 66 % del tiempo y el decoder original 33 %. El
denoiser INT8 mejoró 32 %; el decoder `bridge-fast` FP32 mejoró 69.3 % frente al
decoder original. La composición selectiva persistente se midió en 67.5 ms de
mediana y 78.1 ms P95 durante 30 corridas, con 10.52 MB de modelos. La evidencia
queda registrada en `docs/experiments/android-bridge-selective-pipeline-2026-08-13.md`
y ADR 0006.
