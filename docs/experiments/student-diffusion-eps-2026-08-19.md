# Estudiante compacto, difusión con predicción de epsilon — ADR 0010 (2026-08-19)

## Objetivo

Formulación B del ADR 0010: el mismo U-Net de 52,231,267 parámetros entrenado
como predictor de epsilon sobre la release `avatarface-distill-teacher` v1.0.0,
con muestreo DDIM de 8 pasos, tras descartar la formulación directa
(`docs/experiments/student-direct-2026-08-19.md`).

## Resultado: el modelo aprende, el muestreo desde ruido puro no

A los 10,000 pasos (pérdida 0.002, prácticamente convergida) las muestras de
control mostraron el avance decisivo frente a la formulación directa:
**aparecen los rasgos faciales** —gafas, ojos con iris azul, cejas, nariz—,
que A nunca produjo ni sobre datos de entrenamiento. Pero todas las muestras
salen **lavadas en verde/azul**, con el color global equivocado.

El diagnóstico (`diag_sampler.py`, `diag_tmax.py`) separó modelo de
muestreador:

1. **El modelo es excelente.** Partiendo de una imagen real del split train
   con ruido a t=0.5, la reconstrucción de x0 devuelve el avatar completo con
   gafas, ojos, sonrisa y colores correctos, indistinguible del maestro.
2. **La proyección a x0 estalla a ruido alto.** A t=0.99 el error de epsilon
   es mínimo (`eps_mse=0.000174`) pero el rango de x0 se dispara a
   (−49.40, 36.78): dividir por sqrt(ab)≈0.01 amplifica cualquier error ×100.
3. **No es discretización.** Muestrear con 8, 30 o 100 pasos da el mismo
   defecto de color.
4. **No se arregla moviendo el arranque de la cadena.** Con t_max de 0.7 a
   0.97 las imágenes salen planas (std 0.076–0.149 frente a ~0.3 de las
   reales): por debajo de t≈1 la marginal real conserva entre 8 % y 44 % de
   señal, así que inicializar con ruido puro es una entrada fuera de
   distribución.

La causa de fondo no es numérica sino del objetivo: **a ruido alto la
predicción óptima de epsilon es esencialmente la propia entrada ruidosa,
cualquiera que sea el condicionamiento**. El objetivo no entrega gradiente
para usar los atributos en los primeros pasos de la cadena —los que fijan
color y estructura global—, de modo que esos pasos quedan de hecho sin
condicionar. Coincide exactamente con lo observado: formas correctas
(construidas en los pasos tardíos) y color global equivocado.

## Consecuencia: parametrización v

La corrección estándar para muestreo de pocos pasos es la parametrización
**v** (Salimans & Ho), `v = sqrt(ab)·eps − sqrt(1−ab)·x0`: a ruido alto el
objetivo tiende a la imagen limpia, así que el condicionamiento manda desde el
primer paso, y x0 y eps se despejan de v **sin dividir por sqrt(ab)**. La
verificación algebraica local confirma la estabilidad justo donde epsilon
fallaba: a t=0.999 el error de reconstrucción de x0 es 1.19e-07.

`scripts/train_student.py` gana la formulación `vpred` (entrenamiento y
muestreo) conservando `diffusion` para reproducibilidad histórica. La corrida
de epsilon se detuvo en el paso 10,000 en vez de agotar los 50,000: con la
pérdida ya convergida, más pasos no cambian un defecto del objetivo.

Esto **no es una iteración más del mismo mecanismo** en el sentido del punto 4
del ADR 0009: el mecanismo (estudiante compacto sobre salidas del maestro) es
el mismo y está validado —el modelo reconstruye perfectamente—; lo que se
corrige es la parametrización del objetivo de mi propio script.

## Evidencia archivada

`artifacts/student-diffusion-1/`: log, muestras de control del paso 10,000,
reconstrucciones de x0 a t=0.2/0.5/0.8/0.99, muestreos con 8/30/100 pasos y
barrido de t_max, con `SHA256SUMS`. El checkpoint no se conserva.

## Presupuesto

Consumo acumulado de la fase: ~2.5 h de dataset + ~7 h de la formulación
directa + ~5 h de epsilon ≈ 14.5 h de las 40 h del tope del ADR 0010. La
corrida `vpred` (~14 h) deja el total en ~28.5 h, dentro del tope.
