# Pipeline selectivo persistente en TECNO KM5s

## Objetivo

Validar de extremo a extremo la composición elegida después del perfilado por
componentes, sin inferir su latencia mediante una suma de benchmarks aislados.

## Composición

| Componente | Precisión | Artefacto | Tamaño |
|---|---|---|---:|
| Encoder | FP32 | `bridge-encoder` | 2,262,846 B |
| Denoiser | INT8 QDQ preprocesado | `bridge-denoiser-int8-preprocessed` | 8,224,426 B |
| Decoder | FP32 | `bridge-fast-decoder` | 32,747 B |
| Total | Mixta | Tres sesiones persistentes | 10,520,019 B |

El encoder se ejecuta una vez por prompt y su tensor de condicionamiento se
mantiene vivo. Cada corrida medida cubre exclusivamente denoiser→decoder mediante
`session.run`; checksum y acceso al tensor de salida quedan fuera del intervalo.

## Resultados Android

Serie formal de 30 corridas después de un warm-up:

| Métrica | Resultado |
|---|---:|
| Mínimo | 58.6 ms |
| Mediana | 67.5 ms |
| Media | 66.9 ms |
| P90 | 73.1 ms |
| P95 | 78.1 ms |
| Máximo | 78.9 ms |
| Encoder ejecutado una vez | 1.45 ms |
| Creación total de sesiones | 532.2 ms |
| PSS máximo muestreado | 162,859 KiB |

Creación de sesiones:

- encoder: 42.9 ms;
- denoiser: 483.8 ms;
- decoder: 5.6 ms.

El denoiser INT8 domina el arranque. Las sesiones deben cargarse de forma
anticipada y mantenerse durante la vida útil de la pantalla de generación.

## Comparación

| Pipeline | Mediana CPU |
|---|---:|
| `bridge` FP32 | 115.5 ms |
| `bridge` INT8 completo | 80.2 ms |
| `bridge-fast` FP32 integrado | 84.8 ms |
| Selectivo persistente | 67.5 ms |

La composición selectiva mejora 41.6 % frente a `bridge` FP32, 15.8 % frente al
INT8 completo y 20.3 % frente a `bridge-fast` FP32.

## Estabilidad térmica

- Thermal Status: 0.
- CPU/GPU/NPU/SoC informados por HAL: 27.6 °C.
- Batería: 22 °C, 100 %, conectada por USB.
- No se observó throttling.

## Decisión

Este pipeline pasa a ser el baseline móvil provisional para el diseño del modelo
entrenable. La compuerta siguiente no es aumentar parámetros: es reemplazar la
calibración sintética del denoiser por datos representativos y comprobar calidad
visual del decoder fast después de entrenamiento.
