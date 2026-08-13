# Registro de riesgos

Escala: probabilidad e impacto de 1 a 5. La exposición es su producto.

| ID | Riesgo | P | I | Exposición | Mitigación | Estado |
|---|---|---:|---:|---:|---|---|
| R-01 | Componentes del modelo con términos incompatibles | 4 | 5 | 20 | Auditoría por componente y alternativa propia | Abierto |
| R-02 | Dataset sin derechos suficientes | 4 | 5 | 20 | Manifiesto obligatorio y compuerta del 100 % | Abierto |
| R-03 | Modelo demasiado grande para Android | 4 | 5 | 20 | Estudiante pequeño, pocos pasos y presupuesto por componente | Abierto |
| R-04 | Operadores no exportables | 4 | 4 | 16 | Exportación temprana antes de corrida extensa | Abierto |
| R-05 | INT4 degrada calidad o no acelera | 4 | 4 | 16 | INT8 estable y precisión mixta por sensibilidad | Abierto |
| R-06 | Fallback silencioso a CPU | 3 | 4 | 12 | Instrumentar backend y latencia por etapa | Abierto |
| R-07 | Throttling térmico | 3 | 4 | 12 | Pruebas repetidas en teléfono físico | Abierto |
| R-08 | Similitud no deseada con personas reales | 3 | 5 | 15 | Datos sintéticos, deduplicación y evaluación facial | Abierto |
| R-09 | Sesgos en atributos faciales | 4 | 4 | 16 | Balance, métricas por grupo y revisión humana | Abierto |
| R-10 | Pérdida de corrida Vast.ai | 3 | 5 | 15 | Checkpoints y respaldo externo verificado | Abierto |
| R-11 | Costos GPU sin mejora | 3 | 4 | 12 | Smoke tests, early stopping y presupuesto por corrida | Abierto |
| R-12 | Falta de Python compatible local | 5 | 2 | 10 | Instalar Python 3.12 aislado antes de ML | Detectado |
| R-13 | El dispositivo físico puede no estar disponible durante una prueba | 2 | 2 | 4 | TECNO KM5s detectado; scripts con serial explícito | Mitigado |
| R-15 | Presión de memoria en un teléfono con 3.66 GiB | 4 | 5 | 20 | Presupuesto ≤1 GiB y benchmark de varias generaciones | Abierto |
| R-14 | Java/Gradle incompatibles con Android | 3 | 3 | 9 | Gradle Wrapper y toolchain JDK fijado | Abierto |

El registro se actualizará cuando cambie una exposición o una mitigación tenga
evidencia. Los riesgos legales y de exportabilidad son bloqueantes.
