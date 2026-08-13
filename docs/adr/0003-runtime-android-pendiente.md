# ADR-0003: Seleccionar el runtime Android mediante benchmark

- Estado: propuesta
- Fecha: 2026-08-13

## Contexto

ExecuTorch, ONNX Runtime Mobile y NNAPI tienen diferencias de operadores,
tamaño, aceleración y compatibilidad por dispositivo. Elegir antes de exportar
un modelo mínimo produciría un compromiso sin evidencia.

## Decisión propuesta

Exportar el mismo modelo pequeño, ejecutarlo en el mismo teléfono y comparar:

- operadores soportados y fallbacks;
- tamaño adicional en la aplicación;
- carga, memoria y latencia;
- estabilidad y complejidad de integración.

## Consecuencias

El proyecto Android completo se posterga hasta obtener este benchmark. Se
mantiene únicamente su contrato y directorio de integración.
