# Selección preliminar del modelo

## Decisión actual

No descargar todavía ningún modelo completo. La primera arquitectura de
producción será un estudiante especializado en avatares y diseñado para
Android. Los modelos grandes sólo podrán actuar como maestros si superan la
compuerta legal y el costo de la prueba está justificado.

## Shortlist

### 1. Estudiante propio — ruta recomendada

Objetivo inicial:

| Componente | Presupuesto orientativo |
|---|---:|
| Encoder de texto | 20–60 M parámetros |
| Denoiser | 100–180 M parámetros |
| Autoencoder | 15–40 M parámetros |
| Total | 135–280 M parámetros |

El modelo generará a 256 × 256 y utilizará como máximo cuatro pasos. La
arquitectura se limitará desde el comienzo a operaciones exportables.

Ventajas:

- pesos resultantes bajo control del proyecto;
- especialización fuerte en rostros de avatar;
- presupuesto Android por componente;
- encoder de texto pequeño;
- cuantización y destilación incorporadas al diseño.

Riesgos:

- mayor necesidad de datos y entrenamiento;
- un modelo entrenado desde cero puede requerir más GPU de la prevista;
- calidad dependiente de un dataset especializado suficiente.

### 2. FLUX.1-schnell — posible maestro

Ventajas: Apache-2.0 declarado, pocos pasos y buena capacidad general.

Desventajas: checkpoint aproximado de 23.8 GB, encoder grande, acceso gated y
costo alto. No se distribuirá ni intentará ejecutar en Android.

### 3. Würstchen v2 — posible maestro o referencia

Ventajas: MIT declarado y compresión espacial extrema.

Desventajas: pipeline aproximado de 11.4 GiB, encoder CLIP-bigG y limitaciones
de reconstrucción facial documentadas. Antes de usarlo deben fijarse todos sus
componentes.

## Referencias no utilizables como base completa

- SANA: estudiar DiT lineal, DCAE y cuantización; no usar Gemma/pipeline.
- SANA-Sprint: estudiar consistencia de un paso; no usar el pipeline.
- MobileDiffusion y SnapFusion: estudiar bloques y reducción de pasos.
- MixDQ: estudiar asignación de precisión según sensibilidad.

Estudiar una idea de un paper o arquitectura no implica copiar pesos o código
con términos incompatibles.

## Próximo experimento

Antes de descargar un maestro se construirá un `mobile feasibility model` con
pesos aleatorios:

- encoder de texto mínimo;
- denoiser de 100–150 M o una escala menor representativa;
- autoencoder pequeño;
- resolución 256;
- exportación a los runtimes Android candidatos.

Este modelo no mide calidad visual. Mide exportabilidad, tamaño, RAM, latencia
de operadores y viabilidad del presupuesto. Si falla en el TECNO KM5s, se
ajusta la arquitectura sin haber pagado entrenamiento.

## Criterio para autorizar una descarga

Una descarga requiere:

1. revisión exacta fijada;
2. auditoría de todos los componentes;
3. tamaño conocido y espacio temporal presupuestado;
4. objetivo experimental explícito;
5. hash esperado o manifiesto reproducible;
6. procedimiento de empaquetado, Drive y limpieza;
7. límite de costo de Vast.ai.
