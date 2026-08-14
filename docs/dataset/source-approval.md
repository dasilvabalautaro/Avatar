# Fuentes especializadas para avatares

Esta es la compuerta de admisión del corpus de entrenamiento. La aprobación de
una fuente no sustituye la evidencia por activo: cada imagen mantiene creador,
licencia, URL de licencia, cesión/consentimiento, hash y split en el manifiesto.

| ID | Fuente | Estado | Decisión |
|---|---|---|---|
| `AF-PROC-001` | Avatares procedimentales propios | Aprobada | Puede entrar ahora; CC0, sin personas ni assets externos. |
| `AF-COMM-001` | Ilustración original por encargo | Condicional | Requiere cesión escrita por activo antes de cualquier ingestión. |
| `AF-CONSENT-001` | Aportes voluntarios de arte de avatar | Condicional | Requiere release, autoridad/edad y trazabilidad por activo. |

No están aprobadas las fotografías de rostros, scraping de galerías/redes,
datasets con licencia o procedencia incompleta, ni imágenes generadas por un
modelo tercero sin confirmar explícitamente que permite destilación,
fine-tuning y redistribución. Las fuentes condicionales **no** se pueden
declarar aprobadas ni incluir en un manifiesto hasta que exista la evidencia
indicada en [`configs/dataset-sources.json`](../../configs/dataset-sources.json).

La primera release ampliada usa exclusivamente `AF-PROC-001`. Es jurídicamente
limpia y verificable, pero sigue siendo un corpus de estilo procedimental: no
equivale a variedad artística suficiente para un entrenamiento de producto. La
entrada de una fuente condicional exige revisión manual documentada.
