# Perfil bridge e INT8 en TECNO KM5s

## Objetivo

Cubrir el salto entre `micro` (0.3 M de parámetros) y `target` (136 M) con un
grafo que haga visible el costo de los pesos y la cuantización sin comprometer la
memoria del teléfono.

## Perfil

- 8,238,551 parámetros reales.
- 6 bloques, ancho 320, 8 cabezas y 2 pasos.
- Imagen de salida: 256 × 256.
- Pesos aleatorios; no evalúa calidad visual.

## Validación local

| Artefacto | Tamaño | Reducción | Error abs. medio | SHA-256 |
|---|---:|---:|---:|---|
| FP32 | 33,001,082 B | — | — | `f5ad44e307ecc1965841c13ac3bbbd22832ddbd761aad0cf80d9323c665ba4db` |
| INT8 directo | 10,296,525 B | 68.8 % | 0.000251 | `9798654a80fa82206d9e577004fa5931c70a37838fa935aa6eb2a11f9d3c5f10` |
| INT8 preprocesado | 10,649,062 B | 67.7 % | 0.000256 | `98460e34b30fb80c2c6650c9ad149a49b3c7c484a9861c0ece431cd8bde179f9` |

El preprocesamiento aplica inferencia de shapes y optimización ONNX Runtime antes
de calibrar. Se incluyó `Gemm` en la política porque la optimización fusionó
varios pares `MatMul+Add`.

## Android CPU

Siete corridas después de un warm-up:

| Artefacto | Sesión | Mediana válida | PSS máximo muestreado |
|---|---:|---:|---:|
| FP32 | 467.2 ms | 115.5 ms | 205,482 KiB |
| INT8 preprocesado | 715.5 ms | 80.2 ms | 165,057 KiB |

La variante INT8 redujo la mediana CPU 30.6 % y el pico PSS muestreado 19.7 %
frente a FP32. La creación de sesión empeoró 53.1 %, por lo que el modelo deberá
cargarse una vez y reutilizarse.

Estas cifras sustituyen las primeras corridas, que incluían el checksum RGB en el
intervalo cronometrado.

## Android NNAPI

- Las latencias iniciales FP32/INT8 de NNAPI incluían checksum y no se conservan
  como métricas válidas.
- Una política inicial INT8 limitada a `Conv` y `MatMul` produjo 111 particiones;
  648 de 1,098 nodos fueron soportados tras optimización. Su latencia histórica
  queda invalidada por el problema de cronometraje.
- Las variantes que incorporan `Gemm`, directa y preprocesada, no completaron el
  benchmark dentro de 120 segundos y fueron detenidas automáticamente.
- El teléfono terminó con estado térmico 0 y 27.9 °C según HAL; el timeout no se
  atribuye a throttling.

## Decisión

1. ONNX Runtime CPU + INT8 preprocesado es la ruta móvil provisional.
2. NNAPI queda deshabilitado para esta arquitectura; no es aceptable depender de
   un fallback fragmentado.
3. `target` no se desplegará todavía. Primero se perfilarán por separado encoder,
   denoiser y decoder para identificar el componente dominante.
4. La sesión se mantendrá viva entre generaciones para amortizar su creación.
5. Los JSON previos se eliminan en el dispositivo antes de cada corrida y la app
   se detiene en un bloque de limpieza incluso ante timeout.
