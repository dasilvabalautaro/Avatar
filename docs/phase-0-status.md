# Estado de la Fase 0

## Resultado

La base de la Fase 0 está implementada y verificada el 13 de agosto de 2026.

## Completado

- Repositorio Git inicializado.
- Estructura Python `src/` con Clean Architecture.
- Entidad y validación inicial de prompts.
- Puerto Android y adaptador ADB.
- CLI ejecutable con `status` y `validate-prompt`.
- Configuración base de proyecto y objetivos medibles.
- Pruebas unitarias.
- Requisitos de producto y alcance Android.
- Inventario del TECNO KM5s conectado por USB.
- Registro de riesgos.
- ADR de arquitectura, Android y licencias.
- Backlog de optimización.
- Exclusión de datos, modelos, descargas, paquetes y secretos de Git.

## Evidencia

- pytest: 10 pruebas superadas.
- Ruff: sin hallazgos.
- mypy estricto: 12 archivos fuente sin errores.
- compileall: correcto.
- CLI de validación: correcta.
- CLI de entorno: ADB disponible y un TECNO KM5s autorizado.

## Decisiones obtenidas de evidencia

- Android 15/API 35 y `arm64-v8a` forman el objetivo físico inicial.
- El teléfono informa aproximadamente 3.66 GiB de RAM.
- El presupuesto máximo del proceso se reduce a 1.0 GiB.
- Los sensores térmicos permiten observar CPU, GPU, NPU y SoC.
- La selección del runtime continúa pendiente de benchmark real.

## Acciones antes del trabajo de ML

1. Confirmar el estilo visual inicial y los idiomas del MVP.
2. Implementar el benchmark temprano de viabilidad Android.

Python 3.12.14, `.venv` y el lockfile de desarrollo quedaron completados al
comenzar la Fase 1.

El type-check de esta fase se ejecutó desde un entorno temporal sin caché; ese
entorno se elimina al finalizar la verificación.
