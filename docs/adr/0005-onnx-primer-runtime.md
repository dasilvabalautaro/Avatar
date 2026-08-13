# ADR-0005: ONNX Runtime para el primer vertical slice

- Estado: aceptada
- Fecha: 2026-08-13

## Contexto

La máquina de desarrollo es macOS Intel. ExecuTorch 1.3 soporta sus herramientas
host en macOS ARM64; Intel requiere compilar PyTorch desde fuente. ONNX Runtime
publica herramientas para Mac x64 y un AAR Android con CPU y NNAPI.

## Decisión

Usar ONNX Runtime 1.23.2 para la primera exportación y prueba en el TECNO KM5s.
Comparar CPU y NNAPI. ExecuTorch permanece como candidato posterior y podrá
exportarse desde una instancia Linux en Vast.ai.

## Consecuencias

- La decisión no selecciona todavía el runtime final.
- El primer APK incorpora el AAR completo, no un runtime reducido.
- El modelo usa operadores ONNX estándar y opset 17.
- Una futura comparación debe usar el mismo perfil y precisión.
