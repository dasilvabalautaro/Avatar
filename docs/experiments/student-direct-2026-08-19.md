# Estudiante compacto, formulación directa — etapa E3 del ADR 0010 (2026-08-19)

## Objetivo

Entrenar el estudiante de 52,231,267 parámetros con la **formulación A
(directa)**: (ruido, atributos) → imagen del maestro, pérdida L1, inferencia
en 1 paso (`docs/student-distill-design.md`).

## Configuración

- Dataset: release `avatarface-distill-teacher` v1.0.0 (3276 muestras train),
  manifiesto SHA-256 `05f36cb1…`.
- `scripts/train_student.py --formulation direct --batch-size 16`, lr 1e-4
  coseno con calentamiento, EMA 0.999, semilla 42, RTX 4090.
- Batch 32 en fp32 **excede la VRAM** (pedía 46 GiB de 47.38); con 16 el uso
  estable es de 40.4 GiB. Queda registrado para futuras corridas.

## Resultado: compuerta no superada; corrida detenida en el paso 25,000

La pérdida bajó de forma sana (0.48 → 0.23 en 5k → 0.17 en 19k → 0.16 en 25k)
y las muestras de control aprendieron **toda la estructura de baja
frecuencia**: silueta de la cabeza, forma y color del pelo, tono de piel y
color de fondo correctos por atributo, incluso en captions de validación no
vistos. Pero **ningún rasgo facial** (ojos, cejas, boca, gafas) apareció entre
los pasos 5,000 y 25,000.

El diagnóstico decisivo (`diag_student.py`, paso 25,000) reconstruyó captions
del **split train** con los pesos crudos y con los EMA:

- la imagen del maestro para `avatar-00002` tiene gafas, ojos y sonrisa;
- la reconstrucción del estudiante para **esa misma muestra de entrenamiento**
  es un rostro liso, sin ningún rasgo;
- pesos crudos y EMA dan el mismo resultado.

No es una brecha de generalización: el modelo **no ajusta ni siquiera los
datos de entrenamiento** tras ~122 épocas. El patrón es el propio de la
regresión L1 con entrada de ruido puro: la estructura global se sintetiza
desde el condicionamiento en el cuello de botella, mientras las conexiones de
salto llevan a la resolución alta el ruido de entrada, que no aporta señal;
las altas frecuencias se promedian y desaparecen. Añadir pérdida perceptual
(la mitigación prevista en el diseño) atacaría el síntoma, no la causa.

Continuar los 25,000 pasos restantes costaba ~7 h de GPU con la curva ya
aplanada y sin cambio de régimen esperable, así que la corrida se detuvo y se
pasó directamente a la **formulación B (difusión)**, el respaldo designado en
el punto 4 del ADR 0010. En difusión la entrada ruidosa **sí** contiene señal
en cada paso, por lo que las conexiones de salto dejan de ser un lastre y el
objetivo de predicción de epsilon no promedia las altas frecuencias.

## Evidencia archivada

`artifacts/student-direct-1/`: log completo, muestras de control del paso
25,000 y las imágenes del diagnóstico (maestro vs. estudiante vs. EMA, train y
validation), con `SHA256SUMS`. El checkpoint no se conserva: sin valor de
continuidad, conforme al criterio de las etapas de destilación previas.

## Consecuencia

Formulación A descartada con evidencia. La iteración única de la formulación B
queda lanzada en la misma instancia (`artifacts/student-diffusion-1`, 50,000
pasos, batch 16, misma semilla y dataset). El presupuesto de la fase sigue
holgado: ~2.5 h de dataset + ~7 h de A + ~14 h de B ≈ 24 h de las 40 h del
tope del ADR 0010.
