# Perfilado por componentes del perfil bridge

## Objetivo

Separar encoder, denoiser y decoder para identificar el costo dominante del
pipeline y decidir dónde aporta valor INT8. Todos los grafos conservan los pesos
deterministas del modelo `bridge`; siguen siendo pesos aleatorios.

## Distribución del modelo

| Componente | Parámetros | Proporción | FP32 ONNX |
|---|---:|---:|---:|
| Encoder | 565,568 | 6.9 % | 2,262,846 B |
| Denoiser | 7,422,404 | 90.1 % | 29,730,040 B |
| Decoder | 250,579 | 3.0 % | 1,007,397 B |

La suma coincide con los 8,238,551 parámetros del pipeline completo.

## Android CPU

Diez corridas después de un warm-up, TECNO KM5s y ONNX Runtime 1.23.2:

| Componente | FP32 | INT8 preprocesado | Cambio | Sesión FP32 | Sesión INT8 |
|---|---:|---:|---:|---:|---:|
| Encoder | 1.3 ms | No medido | — | 34.1 ms | — |
| Denoiser | 80.6 ms | 54.8 ms | −32.0 % | 383.6 ms | 523.1 ms |
| Decoder original | 39.6 ms | 28.9 ms | −27.0 % | 61.5 ms | 104.0 ms |
| Decoder slim | 15.3 ms | 16.5 ms | +8.0 % | 107.1 ms | — |
| Decoder fast | 12.1 ms | 14.0 ms | +15.5 % | 31.1 ms | — |

La suma encoder FP32 + denoiser INT8 + decoder original INT8 es aproximadamente
85.0 ms, coherente con los 80.2 ms del grafo completo INT8 preprocesado. La
composición selectiva con decoder fast FP32 fue medida finalmente en 67.5 ms.

Las primeras cifras del decoder incluían por error el checksum RGB dentro del
cronómetro. El perfil por operador permitió detectarlo; todas las cifras de esta
tabla fueron repetidas midiendo exclusivamente `session.run`.

## Tamaño y error local de INT8

| Componente | Reducción | Error absoluto medio | Decisión |
|---|---:|---:|---|
| Encoder | 5.2 % | 0.01636 | Mantener FP32 inicialmente |
| Denoiser | 72.3 % | 0.06397 | INT8 condicionado a calibración real |
| Decoder | 72.6 % | 0.000299 | INT8 aprobado técnicamente |

El error del denoiser no es una métrica perceptual y la calibración usa tensores
sintéticos. Antes de entrenar, su política INT8 deberá recalibrarse con latentes y
condicionamientos representativos.

## Memoria observada

- Denoiser: PSS máximo muestreado bajó de 205,013 a 129,381 KiB.
- Decoder original: PSS máximo muestreado subió de 121,875 a 130,069 KiB.

Los procesos se midieron por separado y el PSS incluye runtime; estas cifras no
se suman para estimar el pipeline completo. INT8 aporta una reducción clara en el
denoiser, pero no en el decoder.

## Conclusión arquitectónica

El denoiser original consume aproximadamente 66 % del tiempo aislado y el decoder
original 33 %. Reducir el decoder a tres escalas lo llevó a 12.1 ms, 69.3 % menos
que el original. INT8 deja de ser conveniente cuando el decoder ya es pequeño.

Prioridades siguientes:

1. Mantener encoder FP32 y cachear su salida por prompt.
2. Usar INT8 en el denoiser después de calibración representativa.
3. Adoptar `bridge-fast` FP32 como decoder experimental; INT8 no aporta latencia.
4. Mantener las tres sesiones vivas para amortizar su creación acumulada.
5. Evaluar visualmente bordes, ojos y cabello antes de aceptar un decoder menor.
