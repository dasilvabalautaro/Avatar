# Estudiante ligero: compuerta superada y muro de cuantización (2026-08-20)

Etapa de entrenamiento y medición del ADR 0011.

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

## Compuerta del ADR 0011: **superada**

Muestras de control del paso 50,000, generadas con **4 pasos DDIM** (el
contrato móvil) sobre captions del split validation:

- **rostros válidos: 8/8**; **adultos: 8/8**;
- fondo 8/8, tono de piel 8/8, color de pelo 7/8 (`avatar-00051` sale lavado);
- `avatar-00041` acierta el pelo rosa que el modelo de 52 M rendía violeta;
- `avatar-00061` reproduce el desvío del propio maestro (pelo castaño y gafas
  normales donde el caption pedía rosa y gafas de sol).

El acabado es más «pincelado» que el del modelo de 52 M y aparece alguna
asimetría ocular, pero los rostros son válidos. Con **3 pasos** también se
obtienen rostros válidos, aunque más blandos y perdiendo las gafas de
`avatar-00031`; 4 pasos es el punto correcto.

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

Medidas y razonadas, en orden de relación coste/beneficio:

1. **Resolución interna de 128 px con reescalado a 256** — la evidencia del
   ADR 0011 muestra que el coste móvil lo dominan las activaciones, no los
   pesos, así que dividir por cuatro los píxeles debería llevar fp32 a ~2.8 s,
   **con margen y sin cuantizar**. El dominio (arte vectorial plano) tolera
   bien el reescalado. Exige reentrenar (~3–4 h de GPU).
2. **Entrenamiento consciente de cuantización (QAT)** — haría viable el INT8
   de 4.46 s conservando calidad, pero requiere implementación nueva y GPU.
3. **2 pasos en fp32** — 5.5 s, sigue por encima y además degrada la calidad.

La 1 es la recomendada. Elegir entre ellas es decisión del usuario, con su
propio ADR y presupuesto.

## Artefactos

`artifacts/student-lite-1/`: checkpoint, log, muestras de control de 4 y 3
pasos, ONNX fp32 y las cuatro variantes cuantizadas, con `SHA256SUMS`.
Resultados de dispositivo en `artifacts/android/`.
