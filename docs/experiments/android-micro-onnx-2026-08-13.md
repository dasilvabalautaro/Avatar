# ONNX micro en TECNO KM5s

## Objetivo

Validar el recorrido PyTorch → ONNX → APK → ONNX Runtime Android antes de
entrenar o descargar pesos. El grafo contiene pesos aleatorios y no mide calidad
visual.

## Entorno

- Fecha: 2026-08-13.
- Dispositivo: TECNO KM5s.
- Android: 15, API 35.
- ABI: `arm64-v8a`.
- SoC informado: MT6769.
- RAM física: aproximadamente 3.66 GiB.
- Runtime: ONNX Runtime Android 1.23.2.
- Modelo: perfil `micro`, opset 17.
- Parámetros: 304,415.
- ONNX: 1,235,972 bytes.
- SHA-256 ONNX: `a0b0ac0af72820a9296f8f11cc68bbb7ac42aa61612db7312c408514bd996dbe`.
- Corridas medidas: siete, después de un warm-up.
- CPU ORT: cuatro threads.
- APK debug inicial con dos ABI: aproximadamente 78 MB.
- APK debug limitado a `arm64-v8a`: aproximadamente 21 MB; SHA-256
  `e2643554fef1f0ac05adee44e969a6d165f981d1d1a48c8df92b827c68e0efb7`.

## Validación local

- PyTorch: 2.2.2.
- ONNX: 1.22.0.
- ONNX Runtime: 1.23.2.
- Error absoluto máximo PyTorch↔ORT: `8.195638656616211e-08`.
- Tiempo de exportación: 0.286 s.

## Resultados Android

| Métrica | CPU | NNAPI solicitado |
|---|---:|---:|
| Creación de sesión | 150.8 ms | 218.8 ms |
| Mediana de inferencia | 19.7 ms | 153.5 ms |
| PSS máximo muestreado | 111,098 KiB | 115,626 KiB |
| Checksum | 4042.535947 | 4042.536256 |

NNAPI fue aproximadamente 7.8 veces más lento que CPU en la mediana. La diferencia
relativa de checksum es pequeña y esperable por cambios numéricos del backend.

Estas cifras sustituyen la primera medición, que incluía por error el cálculo de
checksum de 196,608 valores dentro del intervalo de inferencia.

## Asignación NNAPI

Los logs de ONNX Runtime informaron dos etapas de optimización/asignación:

- 87 de 99 nodos soportados;
- 100 de 112 nodos soportados;
- ocho particiones NNAPI;
- algunos nodos asignados a CPU.

Por tanto, NNAPI sí intervino, pero el grafo quedó fragmentado. En un modelo tan
pequeño, el overhead de partición y transferencia domina el cómputo.

## Estado térmico posterior

- Thermal Status: 0.
- CPU/GPU/NPU/SoC reportados por HAL: 27.9 °C.
- Batería: 21 °C y 99 %, con USB conectado.
- No se observó throttling.

Estas condiciones no permiten medir energía porque el teléfono estaba conectado
por USB y cargando.

## Conclusión

El vertical slice es viable y estable. CPU es el baseline actual para el perfil
micro. NNAPI no debe descartarse todavía: la comparación relevante será con un
grafo cuantizado y de mayor tamaño, reduciendo particiones y operadores de shape.

El filtro de ABI redujo el APK debug aproximadamente 73 % y el APK resultante se
reinstaló correctamente en el dispositivo. Una validación automatizada posterior
de tres corridas se utilizó para validar el runner, pero quedó sustituida por la
medición corregida anterior.

## Mejoras siguientes

1. Cuantizar el grafo antes de ejecutar el perfil `target`.
2. Medir asignación por operador y evitar fragmentación NNAPI.
3. Separar tamaño del runtime, pesos e instalación en dispositivo.
