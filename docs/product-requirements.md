# Requisitos de producto de AvatarFace

## Estado

- Versión: 0.1
- Etapa: Fase 0
- Plataforma: Android
- Inferencia: local y offline

## Problema

Una aplicación Android necesita convertir una descripción textual en un rostro
de avatar sin depender de un servicio remoto. El modelo debe equilibrar calidad,
control semántico, tamaño, memoria, latencia y licencia.

## Usuario objetivo

Una persona que desea crear una imagen de perfil o personaje mediante una
descripción corta, manteniendo sus prompts e imágenes dentro del dispositivo.

## Propuesta del MVP

El MVP generará un único rostro de avatar estilizado o semirrealista, centrado y
apto para imagen de perfil. No intentará reproducir fielmente una identidad real.

Esta especialización es intencional:

- reduce la capacidad necesaria frente a un generador generalista;
- facilita datasets sintéticos y licenciables;
- disminuye riesgos de identidad y biometría;
- permite medir atributos faciales con una taxonomía acotada.

## Historias principales

1. Como usuario, escribo una descripción y genero un avatar.
2. Como usuario, elijo o reutilizo una seed para repetir un resultado.
3. Como usuario, genero sin conexión a Internet.
4. Como usuario, puedo cancelar una generación en curso.
5. Como evaluador, conozco modelo, runtime, seed y duración de cada resultado.

## Entrada

- Prompt entre 1 y 500 caracteres después de normalización.
- Seed entera entre 0 y `2^32-1`.
- Resolución 256 × 256 obligatoria.
- Resolución 512 × 512 sólo si el dispositivo cumple los límites.

Taxonomía inicial de prompt:

- estilo visual;
- forma del rostro;
- tono de piel;
- cabello y color;
- ojos;
- expresión;
- accesorios;
- vestimenta visible;
- iluminación;
- fondo simple.

## Salida

- Imagen RGB cuadrada.
- Metadatos locales: versión del modelo, runtime, seed, dimensiones y duración.
- Sin telemetría remota obligatoria.

## Requisitos funcionales

- RF-01: generar un avatar desde texto.
- RF-02: aceptar una seed reproducible.
- RF-03: ejecutar sin red.
- RF-04: informar progreso o estado de ejecución.
- RF-05: permitir cancelación segura.
- RF-06: guardar o compartir la imagen mediante mecanismos Android.
- RF-07: registrar métricas de benchmark en builds de prueba.
- RF-08: rechazar entradas vacías o fuera del contrato.

## Requisitos no funcionales

- RNF-01: tamaño objetivo del modelo ≤250 MB y máximo inicial ≤400 MB.
- RNF-02: memoria máxima objetivo ≤1.0 GB en el dispositivo de referencia.
- RNF-03: latencia objetivo ≤5 s a 256 × 256 en el teléfono de referencia.
- RNF-04: estabilidad ≥99 % para prompts válidos del conjunto de regresión.
- RNF-05: cuantización INT8 obligatoria como candidato; INT4 mixta experimental.
- RNF-06: degradación cuantizada ≤5 % en las métricas acordadas frente a FP16.
- RNF-07: sin restricciones de campo de uso en modelos o componentes aprobados.
- RNF-08: procedencia y licencia conocidas para el 100 % del dataset.
- RNF-09: pruebas de rendimiento finales en hardware físico por USB/ADB.

## Seguridad y uso responsable

- El MVP no ofrecerá clonación de identidad.
- El dataset priorizará personajes sintéticos y activos autorizados.
- Se evaluará similitud no deseada con personas reales.
- Se documentarán sesgos de representación y fallos conocidos.
- La aplicación deberá definir una política de prompts y salidas antes de
  distribución pública.

## No objetivos del MVP

- iOS.
- Video o animación.
- Cuerpo completo.
- Creación de celebridades o reproducción exacta de personas.
- Entrenamiento en el teléfono.
- Generación generalista de escenas.

## Preguntas abiertas

- Teléfono Android que será referencia oficial.
- Idiomas iniciales además de español.
- Estilo visual exacto que se priorizará en el dataset.
- Política final de moderación local.
- Forma de distribuir actualizaciones del modelo.
