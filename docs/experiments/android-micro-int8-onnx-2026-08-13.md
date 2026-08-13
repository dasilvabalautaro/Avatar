# ONNX INT8 QDQ micro en TECNO KM5s

> Nota de validez: tamaños y errores locales siguen vigentes. Las latencias
> Android de este informe incluían el checksum dentro del cronómetro y quedan
> sustituidas por benchmarks posteriores; no deben usarse para decidir runtime.

## Objetivo

Comprobar si la cuantización estática INT8 reduce tamaño y mejora el runtime del
grafo de viabilidad antes de aplicarla a un candidato mayor. Tanto los pesos como
los datos de calibración son sintéticos; esta prueba no representa calidad visual.

## Método

- Fuente: `avatarface-feasibility-micro.onnx`, FP32.
- Cuantización: QDQ estática, activaciones QUInt8 y pesos QInt8 por canal.
- Operadores cuantizados: `Conv` y `MatMul`.
- Calibración: 16 entradas sintéticas deterministas.
- Validación local: cuatro entradas distintas a la calibración.
- Android: siete corridas después de un warm-up por backend.
- Dispositivo: TECNO KM5s, Android 15, `arm64-v8a`, SoC MT6769.
- Runtime: ONNX Runtime Android 1.23.2.

## Conversión local

| Métrica | FP32 | INT8 QDQ | Cambio |
|---|---:|---:|---:|
| Tamaño ONNX | 1,235,972 B | 804,383 B | −34.9 % |
| Error absoluto máximo | — | 0.004565 | — |
| Error absoluto medio | — | 0.002167 | — |

SHA-256 INT8:
`9f60ccb0b257bd75c01b63bba81113ebd64dee1b090679013ad3fb4d802182f9`.

La reducción es menor que 4× porque el grafo contiene parámetros, constantes y
operadores que permanecen en punto flotante, además de nodos Q/DQ añadidos.

## Resultado Android histórico — latencia invalidada

| Métrica | FP32 CPU | INT8 CPU | FP32 NNAPI | INT8 NNAPI |
|---|---:|---:|---:|---:|
| Creación de sesión | 87.1 ms | 273.9 ms | 193.8 ms | 218.9 ms |
| Mediana inferencia | 204.0 ms | 206.4 ms | 321.1 ms | 391.0 ms |
| PSS máximo muestreado | 120,958 KiB | 124,598 KiB | 121,908 KiB | 117,512 KiB |

INT8 CPU fue aproximadamente 1.2 % más lento, una diferencia demasiado pequeña
para considerarla material con esta muestra. INT8 con NNAPI fue aproximadamente
21.8 % más lento que FP32 NNAPI y 89.4 % más lento que INT8 CPU.

## Decisión

- La ruta INT8 QDQ queda validada como técnica de reducción de pesos.
- No se afirma una mejora de RAM: el PSS observado incluye runtime y presenta
  variación entre procesos.
- CPU continúa como baseline provisional.
- No se desplegará directamente el perfil `target`: antes se construirá un
  perfil intermedio y se inspeccionará la fragmentación Q/DQ y NNAPI.
- Se comparará preprocesamiento/optimización ONNX previo a cuantización, como
  recomienda la propia herramienta de ONNX Runtime.

## Riesgo detectado y corregido

El primer runner automatizado podía leer un JSON anterior si coincidía el backend.
Se descartó esa lectura y se modificó el criterio para exigir simultáneamente
backend y nombre exacto del modelo antes de aceptar un resultado.
