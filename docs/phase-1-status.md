# Estado de la Fase 1

## Resultado

La primera compuerta documental de modelos y licencias está implementada. No se
descargaron pesos.

## Completado

- Política verificable de licencias permisivas.
- Manifiesto JSON con nueve candidatos.
- Entidades de dominio para componentes y hallazgos.
- Caso de uso de auditoría y repositorio JSON.
- Comando `avatar-face audit-candidates`.
- Matriz documental con evidencia oficial.
- Shortlist técnica y legal preliminar.
- Diseño del benchmark temprano de viabilidad Android.
- Python 3.12.14 instalado.
- `.venv` aislado con dependencias de desarrollo sin caché de descarga.
- Versiones de desarrollo registradas en `requirements-dev.lock`.

## Resultado de selección

- Ningún modelo publicado cabe directamente en el presupuesto Android.
- FLUX.1-schnell pasa la regla automática preliminar, pero su acceso gated debe
  revisarse manualmente antes de descargar.
- Würstchen permanece como maestro potencial pendiente de fijar todos sus
  componentes.
- SANA y SANA-Sprint quedan sólo como referencias arquitectónicas debido al
  encoder Gemma.
- SDXL, PixArt-α y DeepFloyd IF quedan excluidos por licencia.
- AuraFlow y Kandinsky quedan fuera del primer ciclo por tamaño y complejidad.
- La ruta recomendada es un estudiante propio especializado.

## Evidencia de calidad

- pytest: 14 pruebas superadas.
- Ruff: sin hallazgos.
- mypy estricto: 15 archivos fuente sin errores.
- CLI instalada en `.venv` y manifiesto auditado correctamente.

## Siguiente compuerta

Implementar el `mobile feasibility model` con pesos aleatorios y perfiles
`micro`, `target` y `stress`. Antes de incorporar PyTorch se fijará una matriz de
versiones compatible con Python 3.12, exportador Android y runtime elegido para
la prueba.
