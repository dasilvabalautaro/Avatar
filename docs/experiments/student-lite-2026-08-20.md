# Estudiante ligero: compuerta de calidad no superada y muro de cuantización (2026-08-20)

Etapa de entrenamiento y medición del ADR 0011.

**Resultado: la compuerta de calidad NO se supera.** La primera evaluación la
dio por superada; fue un error de criterio, corregido más abajo.

## Entrenamiento

- `scripts/train_student.py --formulation vpred --base-channels 32
  --attention-resolutions 16 --ddim-steps 4 --batch-size 32`, 50,000 pasos,
  lr 1e-4 coseno, EMA 0.999, semilla 42, RTX 4090.
- Dataset: release `avatarface-distill-teacher` v1.0.0 (manifiesto
  `05f36cb1…`), restaurada y verificada en la instancia.
- 7,498,787 parámetros; pérdida final 0.020186; corrida limpia (~4 h).
- La subida del dataset por `scp` se cortaba a mitad (107 MB de 154 MB). Se
  resolvió troceando en 8 partes de 20 MB con verificación de tamaño y
  reintentos; el tar recompuesto verificó su SHA-256. Queda como técnica para
  enlaces inestables, sin necesidad de pasar por Drive.
- No se descargaron los pesos de Würstchen: esta corrida entrena sólo sobre el
  dataset congelado, así que se omitió el bootstrap completo (~10 GB menos).

## Compuerta del ADR 0011: **NO superada** (corregido el 2026-08-20)

Registro de la corrección: en la primera evaluación se dio la compuerta por
superada con 8/8. **Esa evaluación fue incorrecta** y el usuario la rechazó al
ver las imágenes. El criterio aplicado fue «¿es un rostro de adulto
reconocible con los atributos correctos?», y con ese criterio pasa; pero el
estándar del producto es **arte vectorial plano al estilo del maestro**, y
comparadas lado a lado con las salidas del maestro para los mismos captions
las muestras del estudiante ligero **no lo cumplen**:

- bordes difuminados, con aspecto de pintura al óleo en vez de vector plano;
- regiones de color sin planos limpios;
- rasgos deformados: ojos asimétricos en `avatar-00021` y `avatar-00031`;
- gafas resueltas como manchas, no como trazos.

Lo que el modelo **sí** acierta son los atributos globales: fondo 8/8, tono de
piel 8/8, color de pelo 7/8, y `avatar-00041` mejora incluso al modelo de 52 M
(pelo rosa correcto). Pero la fidelidad de atributos no basta: sin nitidez de
estilo el artefacto no es utilizable como producto.

Lección de método: la compuerta de los ADR 0010 y 0011 dice «rostros válidos» y
«fidelidad comparable al maestro», y se evaluó sólo la segunda mitad. Toda
evaluación futura debe hacerse **lado a lado contra la salida del maestro para
el mismo caption**, y juzgar también nitidez, limpieza de bordes y simetría,
no sólo la presencia de los atributos.

### Escala de calidad observada

| Origen | Nitidez | Veredicto |
|---|---|---|
| Maestro (Würstchen + LoRA) | vector plano nítido | referencia |
| Estudiante 52 M, 8 pasos | bordes casi limpios, algo blando | cercano, aún no igual |
| Estudiante 7.5 M, 4 pasos | difuso y deformado | **no aceptable** |
| Generador procedimental (Pillow) | perfectamente nítido | estilo mucho más tosco |

## Medición en dispositivo (TECNO KM5s, ONNX Runtime 1.23.2, CPU)

Ajuste previo: el arnés usaba 4 hilos y el MT6769 tiene 8 núcleos. Medidos
4, 8 y 6 hilos, **6 es el óptimo** (1277 → 1212 → 1114 ms); el APK queda con
`CPU_THREADS = 6`.

| Precisión | ms/paso | 4 pasos | Tamaño | Memoria | Degradación | Imagen |
|---|---|---|---|---|---|---|
| FP32 | 2763 | 11.05 s | 30.2 MB | 0.47 GiB | — (referencia) | válida |
| INT8 (activaciones 8 bits, MinMax) | 1114 | **4.46 s** | 8.4 MB | 0.34 GiB | 24.5 % | **ruido** |
| INT8 selectiva (condicionamiento en fp32) | ≈1114 | ≈4.46 s | 16.5 MB | — | 25.5 % | ruido |
| INT8 percentil | ≈1114 | ≈4.46 s | 8.4 MB | — | 17.7 % | ruido |
| W8A16 (activaciones 16 bits) | 3405 | 13.6 s | 8.5 MB | — | **6.2 %** | válida |

Estado frente a los requisitos:

- **RNF-01 tamaño**: cumple con enorme holgura en todas las variantes
  (8–30 MB frente a 250 MB).
- **RNF-02 memoria**: cumple (0.34–0.47 GiB frente a 1.0 GB).
- **RNF-03 latencia**: **sólo la cumple la variante INT8 de 8 bits** (4.46 s,
  con p90 de 4.71 s).
- **RNF-06 degradación**: esa misma variante degrada un 24.5 % y produce
  ruido; las que preservan la calidad (FP32 y W8A16) están 2.2× y 2.7× por
  encima del presupuesto de latencia.

## Diagnóstico del muro

El modelo de 7.5 M no tolera activaciones de 8 bits: con sólo 4 pasos de
muestreo, el error de cuantización se realimenta en la cadena y el resultado
degenera. Se descartaron por medición dos hipótesis:

1. **No es la vía de condicionamiento**: mantener embeddings, MLP de tiempo,
   FiLM, `stem` y `head` en fp32 (48 nodos de 649) empeoró levemente (25.5 %).
2. **No es sólo la calibración**: pasar de MinMax a percentil bajó de 24.5 % a
   17.7 %, insuficiente.

Lo que sí lo corrige es subir las activaciones a 16 bits (6.2 %), pero ONNX
Runtime no tiene kernels optimizados para 16 bits en CPU ARM y la variante
resulta **más lenta que fp32**.

Conclusión: para este estudiante la cuantización posterior al entrenamiento no
compra latencia sin destruir calidad. La brecha real, con calidad preservada,
es de **2.2×** (11.05 s en fp32 frente a 5 s).

## Palancas restantes

El plan que se había propuesto (**resolución interna de 128 px con reescalado
a 256**) queda **descartado**: atacaba la latencia pero habría empeorado
justo el defecto que hace fallar la compuerta, porque reescalar difumina más
los bordes.

La situación real es una tensión directa entre los dos ejes:

- la **nitidez** exige más capacidad y más pasos de muestreo;
- la **latencia** exige menos de ambos.

Ambos extremos están ahora medidos: 52 M con 8 pasos se acerca al estilo del
maestro pero tarda 55.3 s; 7.5 M con 4 pasos entra en 4.46 s pero produce
imágenes difusas. Ninguna configuración intermedia probada satisface los dos.

Observación estructural, relevante para la decisión: el estudiante está
condicionado **únicamente por los 9 atributos discretos** del vocabulario
cerrado (unas 415,000 combinaciones posibles). No tiene más poder expresivo
que ese vocabulario, así que el modelo funciona como un renderizador caro y
borroso de nueve valores categóricos. El paso texto → atributos ya está
resuelto en código determinista (`domain/attributes.py`), y el paso atributos
→ imagen también existe en código (`infrastructure/dataset/procedural_generator.py`),
con nitidez perfecta por construcción y coste de milisegundos. Lo que aporta
el maestro sobre ese generador es **estética**, no capacidad expresiva: sus
avatares son vectores planos atractivos frente a las formas geométricas toscas
del generador actual.

Opciones a decidir por el usuario, con su propio ADR:

1. **Mejorar el renderizador procedimental** hasta acercarlo a la estética del
   maestro. Es un problema de código puro: nitidez perfecta, milisegundos,
   sin pesos, sin GPU, sin cuantización, offline por construcción. Renuncia a
   la generación neuronal.
2. **QAT** para rescatar el INT8 de 4.46 s conservando la nitidez del modelo
   de 52 M; exige implementación nueva y GPU, y su techo de nitidez sigue sin
   estar demostrado a 4 pasos.
3. **Aceptar un presupuesto de latencia mayor** (revisar RNF-03) y usar un
   modelo intermedio, midiendo antes qué nitidez da.
4. Cambiar el alcance offline (opción (c) del ADR 0009).

## Artefactos

`artifacts/student-lite-1/`: checkpoint, log, muestras de control de 4 y 3
pasos, ONNX fp32 y las cuatro variantes cuantizadas, con `SHA256SUMS`.
Resultados de dispositivo en `artifacts/android/`.
