# Benchmark temprano de viabilidad Android

## Propósito

Comprobar la arquitectura y los operadores antes de descargar datos o ejecutar
entrenamientos costosos. Este benchmark no intenta producir imágenes de calidad.

## Artefacto

Un pipeline con pesos aleatorios y dimensiones configurables:

```text
tokenizer mínimo → encoder de texto → denoiser de pocos pasos
                                      ↓
                              latente 256 × 256
                                      ↓
                               decoder RGB
```

Perfiles:

- `micro`: confirma exportación y carga;
- `target`: aproxima 135–280 M parámetros;
- `stress`: identifica el límite del TECNO KM5s.

## Runtimes

- ExecuTorch.
- ONNX Runtime Mobile.
- NNAPI cuando el runtime permita delegación explícita.

## Mediciones

- porcentaje de operadores exportados;
- operadores con fallback;
- tamaño del grafo y runtime;
- tiempo de carga;
- memoria antes, durante y después;
- latencia por componente y paso;
- cold y warm start;
- diez ejecuciones consecutivas;
- temperatura y estado de throttling;
- determinismo numérico dentro de tolerancia.

## Condiciones

- TECNO KM5s por USB y serial explícito.
- Android 15/API 35.
- Build benchmark/release.
- Batería y temperatura inicial registradas.
- Sin comparar runtimes con grafos o precisiones diferentes sin indicarlo.

## Compuertas

1. El perfil `micro` debe exportar sin operadores personalizados.
2. El perfil `target` debe permanecer por debajo de 1 GiB.
3. No se acepta fallback silencioso.
4. El runtime elegido debe mostrar una ruta viable hacia INT8.
5. Si ningún runtime cumple, se reduce el presupuesto antes de entrenar.

## Salida

Cada ejecución producirá JSON con versión de esquema, commit, dispositivo,
runtime, backend, dimensiones, precisión, métricas y errores, más logs ADB y
hashes de artefactos.
