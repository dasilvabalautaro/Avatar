# ADR 0012: El avatar se dibuja por código, no se genera con un modelo neuronal

- Estado: aceptado.
- Fecha: 2026-08-20.

## Contexto

Tras cerrar la vía de destilación del prior (ADR 0008, 0009) y la del
estudiante compacto (ADR 0010, 0011), la evidencia acumulada es:

- **La calidad visual no se alcanza dentro del presupuesto móvil.** El
  estudiante de 52 M con 8 pasos se acerca al estilo del maestro pero tarda
  55.3 s en el dispositivo; el de 7.5 M con 4 pasos entra en 4.46 s pero
  produce imágenes difusas que el usuario rechazó
  (`docs/experiments/student-lite-2026-08-20.md`).
- **La cuantización no cierra la brecha**: 8 bits produce ruido y 16 bits es
  más lento que fp32 por falta de kernels en CPU ARM.
- **La difusión con pocos pasos es estructuralmente mala para este dominio**:
  el arte vectorial plano son bordes duros y regiones de color sólido, y eso
  es justo lo que un difusor de 4 pasos no resuelve.

El hecho decisivo es de diseño, no de rendimiento: **el modelo estaba
condicionado únicamente por los 9 atributos discretos** del vocabulario
cerrado (unas 415,000 combinaciones). La red no aportaba poder expresivo sobre
el vocabulario; funcionaba como un renderizador caro y borroso de nueve
valores categóricos. Y ambos extremos del recorrido ya existían en código
determinista: texto → atributos (`domain/attributes.py`) y atributos → dibujo
(`infrastructure/dataset/procedural_generator.py`).

## Decisión

1. El producto genera el avatar **dibujándolo por código** desde los
   atributos: `infrastructure/rendering/avatar_renderer.py`
   (`FlatVectorAvatarRenderer`). No hay modelo neuronal en el camino de
   inferencia.
2. El estilo objetivo sigue siendo el del maestro Würstchen + LoRA, que pasa
   de ser el modelo a destilar a ser la **referencia visual** contra la que se
   diseña el dibujo.
3. La nitidez se consigue por construcción: el dibujo se hace a 4× el tamaño
   pedido y se reduce con Lanczos, de modo que los bordes quedan suavizados
   sin perder el plano de color. Las siluetas —rostro, pelo, barba— se
   describen con splines Catmull-Rom y arcos elípticos, no con rectángulos
   redondeados, para que el resultado sea un avatar cuidado y no un montaje
   de primitivas.
4. El filtro de sólo adultos (RF-09) sigue viviendo en `AvatarPrompt`, antes
   del dibujo; el comando `avatar-face render` lo aplica.
5. El trabajo neuronal previo **no se borra**: los ADR 0007 a 0011, sus
   experimentos y la release `avatarface-distill-teacher` v1.0.0 quedan como
   registro y como referencia de estilo.

## Consecuencias

Frente a los requisitos no funcionales, medidos y no estimados:

| Requisito | Presupuesto | Estudiante 7.5 M (INT8) | Dibujo por código |
|---|---|---|---|
| RNF-01 tamaño | ≤ 250 MB | 8.4 MB | **0 MB de pesos** |
| RNF-02 memoria | ≤ 1.0 GB | 0.34 GiB | trivial |
| RNF-03 latencia | ≤ 5 s | 4.46 s (imagen inservible) | **~11 ms** |
| RNF-06 degradación | ≤ 5 % | 24.5 % | **no aplica** |
| Nitidez | estilo del maestro | difusa, rechazada | **exacta por construcción** |

- El alcance offline (ADR 0002) queda garantizado por construcción: no hay
  pesos que descargar ni runtime de inferencia que alimentar.
- La expresividad depende del vocabulario, que por eso se amplió el mismo día
  a **17 atributos y 13,226,976,000,000 combinaciones**: el avatar sustituye a
  una foto de perfil, así que una persona tiene que poder reconocerse en el
  resultado. Los nueve atributos originales se conservan (los manifiestos
  congelados siguen siendo válidos, `LEGACY_ATTRIBUTES`) y los ocho nuevos
  —cejas, nariz, vello facial, gafas, pendientes, pecas, prenda y su color—
  tienen valor por defecto. Ampliar más es trabajo de código, no de GPU.
- La compuerta de licencias y la prohibición de rostros reales se cumplen
  trivialmente: no hay pesos de terceros ni datos de entrenamiento en el
  camino de inferencia.
- El ADR 0006 (cuantización selectiva) y el pipeline ONNX quedan sin uso en
  el camino del producto; se conservan para el registro.
- El estilo es un rasgo de producto, no un detalle: el avatar sustituye a una
  foto de perfil en una app que promete privacidad y ausencia de acoso
  comercial, así que la persona tiene que sentirse tranquila con su
  representación o la descartará. De ahí las reglas de estilo del apartado
  siguiente.
- Trabajo pendiente derivado: llevar el dibujo a la app Android (Kotlin), ya
  sea reimplementando el trazado con las primitivas de Canvas o generando el
  avatar en el lado nativo con la misma tabla de coordenadas.

## Reglas de estilo (revisión del 2026-08-21)

Salieron de comparar los renders con el ojo puesto en «¿me pondría esto como
foto de perfil?». Cualquier cambio de coordenadas debe respetarlas y
comprobarse con `python scripts/render_gallery.py`.

1. **Sin sombreado lateral en el rostro.** La sombra de mejilla que había
   cruzaba la cara con un borde duro y se leía como una mancha. La única
   sombra admitida es la de contacto bajo el mentón, muy tenue.
2. **Nada de negro puro en los rasgos.** Cejas y pestañas se derivan del color
   del pelo; el negro puro endurece el gesto y produce caras severas.
3. **La nariz es la base, no el tabique.** Una media luna fina bajo el
   tabique; una forma grande en mitad del rostro parece una mancha y una línea
   suelta un arañazo.
4. **Las orejas apenas asoman.** Sobresalir más de cuatro píxeles a 256 px las
   convierte en bultos.
5. **Un polígono se recorre en un solo sentido.** Saltar de un lado al otro lo
   cierra sobre sí mismo: así aparecieron la muesca de la coronilla y la barba
   con forma de bufanda.
6. **Capas separadas para barba, boca y bigote**, en ese orden, para que los
   labios queden visibles y el bigote se apoye sobre ellos.
7. **Las siluetas deben distinguirse entre sí.** Si todas las formas de cara
   se parecen, nadie encuentra la suya.
