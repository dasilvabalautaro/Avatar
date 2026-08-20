# ADR 0011: Estudiante ligero con el presupuesto móvil medido en dispositivo

- Estado: aceptado.
- Fecha: 2026-08-20.

## Contexto

El ADR 0010 quedó **validado en calidad**: el estudiante de 52,231,267
parámetros entrenado con parametrización v sobre las salidas del maestro
superó la compuerta con 8/8 rostros válidos y adultos, y sus desvíos de
atributos son los que hereda del propio maestro
(`docs/experiments/student-vpred-2026-08-20.md`).

La etapa E5 midió el artefacto **en el dispositivo de referencia** (TECNO
KM5s, ONNX Runtime 1.23.2, CPU, INT8 QDQ), no por extrapolación:

| Métrica | Presupuesto | Modelo del ADR 0010 |
|---|---|---|
| RNF-01 tamaño | ≤ 250 MB | 53.7 MB — cumple |
| RNF-02 memoria | ≤ 1.0 GB | 0.85 GiB — cumple, justo |
| RNF-03 latencia | ≤ 5 s | 6907 ms/paso × 8 = **55.3 s — no cumple** |
| RNF-06 degradación | ≤ 5 % | 8.4 % — no cumple |

La brecha real es de **11×** en latencia, no los ~38× que sugería la
extrapolación previa: el factor entre la CPU local y el dispositivo resultó
ser 8.9×, no 31×. El tamaño, que era el temor original del ADR 0007, sobra
por 4.7×.

Para elegir la arquitectura sin pagar GPU, se exportaron variantes con **pesos
aleatorios** y se midieron en el dispositivo, igual que hizo el spike de
viabilidad original (los pesos aleatorios validan contratos y rendimiento, no
calidad):

| base_channels | parámetros | INT8 | ms/paso | 4 pasos | memoria |
|---|---|---|---|---|---|
| 96 (entrenado) | 52,231,267 | 53.7 MB | 6907 | 27.6 s | 0.85 GiB |
| 32 | 7,498,787 | 8.4 MB | 1192 | **4.8 s** | 0.34 GiB |
| 24 | 4,941,883 | 5.8 MB | 959 | **3.8 s** | 0.31 GiB |

Hallazgo relevante: entre base 24 y base 32 hay 34 % menos parámetros pero
sólo 20 % menos tiempo. El coste móvil lo dominan los **mapas de activación a
256 px**, no los pesos, así que seguir recortando canales da rendimientos
decrecientes; las palancas eficaces son los pasos de muestreo y la resolución
interna.

## Decisión

1. **Arquitectura del estudiante ligero**: `base_channels = 32`,
   multiplicadores (1, 2, 3, 4), atención **sólo a 16 px** (a 32 px domina el
   coste móvil), 256 px de salida, 7,498,787 parámetros. Es la variante más
   capaz que cabe en el presupuesto medido.
2. **Contrato de inferencia: 4 pasos DDIM** (4.8 s medidos, dentro de
   RNF-03). El número de pasos es una elección de muestreo, no de
   entrenamiento: el modelo se entrena con t continuo, como hasta ahora, y
   `scripts/train_student.py --ddim-steps` fija los pasos de las muestras de
   control y del contrato móvil.
3. **Variante de reserva**: si la cuantización selectiva necesaria para
   RNF-06 consume el 4 % de margen de base 32, se baja a `base_channels = 24`
   (3.8 s medidos, 24 % de margen) o a 3 pasos, en ese orden. No se aceptan
   configuraciones cuyo presupuesto no esté medido en el dispositivo.
4. **Entrenamiento**: misma formulación `vpred`, misma release congelada
   `avatarface-distill-teacher` v1.0.0 (**no se regenera el dataset**), misma
   semilla 42, 50,000 pasos. Al ser ~7× menor, admite batch mayor y se estima
   en 4–6 h de RTX 4090 (~2–3 USD).
5. **Compuertas**: idénticas al ADR 0010 —≥ 6/8 rostros válidos, 8/8 adultos,
   fidelidad comparable al maestro— evaluadas sobre muestras generadas **con
   4 pasos**, no con 8. Además, el artefacto debe cumplir en dispositivo
   RNF-01, RNF-02 y RNF-03, y RNF-06 tras la cuantización selectiva del
   ADR 0006.
6. Si el estudiante ligero supera calidad pero no RNF-06, se aplica la
   cuantización selectiva (mantener en fp32 las capas sensibles) antes de
   considerar cualquier cambio de arquitectura.

## Consecuencias

- `StudentUNet` admite anchos no múltiplos de 32: la cuenta de grupos de
  GroupNorm pasa a ser `gcd(32, canales)`, que **preserva exactamente** el
  valor histórico de 32 para los anchos ya usados (el checkpoint entrenado
  sigue cargando con sus 52,231,267 parámetros).
- `scripts/train_student.py` gana `--attention-resolutions` y `--ddim-steps`.
- El APK instrumental acepta la firma del estudiante (`sample`, `ratio`,
  `attributes`) y trae los modelos de medición como assets.
- El modelo de 52 M del ADR 0010 queda como **referencia de calidad**: no es
  candidato a producto por latencia, pero fija el techo alcanzable.
- Presupuesto: el ADR 0010 consumió ~28 h de sus 40 h. Esta corrida (4–6 h)
  cabe en el remanente sin ampliar el tope.
