# ADR 0006: CPU y cuantización selectiva por componente

- Estado: aceptado.
- Fecha: 2026-08-13.

## Contexto

NNAPI fragmentó el grafo `bridge` y llegó a exceder 120 segundos. INT8 sobre el
pipeline completo mejoró CPU, pero no mostraba qué componente explicaba el costo.

## Decisión

- ONNX Runtime CPU será el backend provisional.
- Encoder permanecerá inicialmente en FP32 y su resultado se cacheará.
- Denoiser utilizará INT8 sólo después de calibración con datos representativos.
- Decoder `bridge-fast` permanecerá FP32: redujo 69.3 % la latencia frente al
  decoder original y su variante INT8 resultó más lenta.
- Las sesiones se crearán una vez y se reutilizarán.

La composición fue validada posteriormente con una mediana de 67.5 ms en 30
corridas y 10.52 MB de modelos.

## Consecuencias

La aplicación tendrá contratos separados para encoder, denoiser y decoder. Esto
facilita calibración, reemplazo y benchmark independiente, a cambio de manejar
tres artefactos y sus versiones compatibles.
