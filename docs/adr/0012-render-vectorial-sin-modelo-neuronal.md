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
- Trabajo pendiente derivado: llevar el dibujo a la app Android (Kotlin), ya
  sea reimplementando el trazado con las primitivas de Canvas o generando el
  avatar en el lado nativo con la misma tabla de coordenadas.
