# Estudiante compacto, parametrización v — compuerta superada (2026-08-20)

## Configuración

- `scripts/train_student.py --formulation vpred --batch-size 16`, 50,000 pasos,
  lr 1e-4 coseno, EMA 0.999, semilla 42, RTX 4090.
- Dataset: release `avatarface-distill-teacher` v1.0.0 (manifiesto SHA-256
  `05f36cb1…`), 3276 muestras de train.
- U-Net de 52,231,267 parámetros a 256 px, condicionado por 9 atributos.
- Pérdida final 0.006306; corrida completa sin incidencias.

## Compuerta del ADR 0010: **superada**

Evaluación sobre las 8 muestras de control del paso 50,000, todas de captions
del split **validation** (nunca vistos en entrenamiento):

- **rostros válidos: 8/8** (la compuerta exige ≥ 6/8);
- **adultos: 8/8** (RF-09);
- fidelidad de atributos **comparable al maestro**: fondo 8/8, tono de piel
  8/8, color de pelo 6/8, accesorios 6/8.

La comparación contra las salidas del propio maestro para esos mismos captions
(las imágenes de la release) es la evidencia decisiva del criterio
«comparable al maestro»:

- `avatar-00061` pedía «pelo rosa corto y gafas de sol»; **el maestro generó
  pelo castaño y gafas normales**, y el estudiante reprodujo fielmente esa
  salida. El desvío es del maestro, heredado como estaba previsto.
- `avatar-00011` (pelo negro, piel morena, fondo arena, pendientes): el
  maestro no dibujó pendientes y el estudiante coincide en todo lo demás.
- Único desvío propio del estudiante: `avatar-00051`, donde el maestro generó
  pelo azul intenso y el estudiante lo produce pálido y lavado.

Un caso de desvío propio está dentro del margen de ±1 de la compuerta.

## Etapa E5: exportación, cuantización y presupuesto móvil

- Exportación ONNX del U-Net (pesos EMA, opset 17, lote fijo 1):
  `avatarface-student.onnx`, 209,169,166 bytes, SHA-256 `85476a16…`. El bucle
  DDIM de 8 pasos queda para la app, como el resto del pipeline móvil.
- Cuantización INT8 QDQ con calibración representativa
  (`scripts/quantize_student_onnx.py`: atributos válidos del vocabulario y
  estados x_t del schedule real; el calibrador genérico del proyecto no sirve
  porque genera índices fuera de rango): **53,736,892 bytes**, SHA-256
  `29f5eee2…`.

Resultado frente a los requisitos no funcionales:

| Requisito | Presupuesto | Medido | Estado |
|---|---|---|---|
| RNF-01 tamaño | ≤ 250 MB | 53.7 MB (INT8) | **cumple con holgura 4.7×** |
| RNF-06 degradación | ≤ 5 % | 8.4 % (MAE de imagen final) | **no cumple** |
| RNF-03 latencia | ≤ 5 s a 256 px | ~192 s estimados | **no cumple, por ~38×** |

Latencia medida en CPU de la máquina local (macOS Intel): 1329 ms por paso en
fp32 y **775 ms en INT8**, es decir 6.2 s por imagen de 8 pasos. La
extrapolación al dispositivo usa el único punto de calibración disponible del
proyecto: el modelo sintético `micro` tarda 2.2 ms aquí y **67.5 ms medidos en
el TECNO KM5s**, un factor de ~31×. Con ese factor, el estudiante en INT8
rondaría los 24 s por paso y ~3.2 minutos por imagen.

La estimación no sustituye a la medición en dispositivo (RNF-09) y el factor
proviene de un modelo mucho menor, así que el número real puede diferir
bastante; pero incluso con un factor conservador de 10× el resultado quedaría
en ~62 s, un orden de magnitud sobre el presupuesto. El teléfono no estaba
conectado durante esta sesión (`avatar-face status`: 0 dispositivos), así que
la medición física queda pendiente.

## Lectura y consecuencia

La vía del ADR 0010 **funciona en calidad**: un estudiante de 52 M entrenado
sobre salidas del maestro genera rostros de avatar adultos válidos con
atributos correctos, algo que ni la formulación directa ni la de epsilon
lograron. El cuello de botella ya no es la calidad ni el tamaño —donde sobra
holgura— sino el **cómputo**: 8 pasos de un U-Net de 52 M a 256 px son
demasiados para una CPU móvil de gama baja.

Las palancas para cerrar la brecha son conocidas y multiplicativas: reducir
los pasos de muestreo (8 → 2 ó 1), reducir la capacidad y la resolución
interna de la red, y aplicar la **cuantización selectiva** del ADR 0006 (que
además es el remedio previsto para la degradación de RNF-06). Elegirlas y
fijar su presupuesto excede lo decidido en el ADR 0010 y corresponde a un ADR
nuevo, con decisión del usuario.

## Artefactos

`artifacts/student-vpred-1/`: checkpoint, log, muestras de control del paso
50,000, ONNX fp32 e INT8 con sus metadatos y hashes.
