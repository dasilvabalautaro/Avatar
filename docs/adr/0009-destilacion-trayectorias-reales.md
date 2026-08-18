# ADR 0009: Destilación del prior sobre trayectorias reales del maestro

- Estado: aceptado.
- Fecha: 2026-08-18.

## Contexto

La destilación progresiva del ADR 0008 falló su compuerta de etapa 1 en tres
formulaciones del objetivo (2026-08-18):

- etapa 1 (peso SNR): 0/8 rostros válidos; el peso anulaba el gradiente en
  saltos de alto ruido (`docs/experiments/wuerstchen-distill-stage-1-2026-08-18.md`);
- etapa 1b (peso uniforme): 1/8; señal restaurada pero 2000 pasos con
  `lr=1e-5` insuficientes (`docs/experiments/wuerstchen-distill-stage-1b-2026-08-18.md`);
- etapa 1c (objetivo normalizado, `lr=5e-5`, 6000 pasos): 0/8; los 8 casos
  producen rostros con rasgos completos pero **repetidos en mosaico**
  (`docs/experiments/wuerstchen-distill-stage-1c-2026-08-18.md`).

El patrón de 1c es la firma de un desajuste de distribución: el estudiante se
entrena sobre la marginal N(0, I) (ruido puro en cualquier timestep) pero en
inferencia recibe sus propios estados intermedios, que no se parecen al ruido
puro. Cada salto «sobre-actúa» y el exceso coherente en los 14 saltos produce
la repetición. Las correcciones baratas del objetivo (pesos, normalización,
lr/pasos) están agotadas; lo que queda es reformular el método.

Opciones evaluadas con el usuario:

- **(a) Reformular la destilación** con trayectorias reales del maestro como
  distribución de entrenamiento.
- **(b) Cuantización agresiva del prior sin reducir pasos**: conserva la
  calidad (RNF-06), pero no cierra la brecha — el prior de 994 M parámetros en
  INT8 (~1 GB) ya supera RNF-01 (máx. 400 MB) y RNF-02 (1.0 GB de memoria), y
  30 pasos de prior + 12 de decoder en CPU móvil superan en órdenes de
  magnitud RNF-03 (≤ 5 s). Queda como complemento posterior (cuantización
  selectiva del artefacto destilado, ADR 0006), no como vía.
- **(c) Aceptar GPU/servidor y replantear el alcance offline**: abandona el
  requisito rector del producto (ADR 0002). Descartada mientras quede una vía
  offline con evidencia.

Aclaración del usuario (2026-08-18) sobre el alcance de dispositivo: el TECNO
KM5s es **sólo el dispositivo de referencia para pruebas y métricas**; el
producto debe funcionar desde **Android API 31 hasta API 36** (Android 12 a
16). Ese rango ya está cubierto (minSdk 26, ONNX Runtime Android 1.23.2, ABI
`arm64-v8a`); los RNF-01 a RNF-03 se siguen verificando en el dispositivo de
referencia, como hasta ahora.

## Decisión

1. **Mecanismo**: destilación progresiva sobre **trayectorias reales del
   maestro**. Fase de generación: el maestro (prior base + LoRA escala-3,
   guía 8.0, rejilla oficial de 30) ejecuta su trayectoria completa para cada
   caption del split train de la release v2.1.0 y se almacenan los latentes en
   todos los puntos de la rejilla. Fase de entrenamiento: cada salto del
   estudiante se entrena desde el latente real x(t) hacia el latente real
   x(t′′) de la misma trayectoria; el maestro no vuelve a ejecutarse durante
   el entrenamiento (menos VRAM y pasos más baratos que en 1-1c).
2. **Una sola variable cambia respecto a la etapa 1c**: la distribución de
   entrada. Se conservan `lr=5e-5`, 6000 pasos, pérdida normalizada
   (mse/‖eps‖²), semilla 42 y la rejilla 30→15, para atribución limpia del
   efecto. La etapa 2 (15→8) usa como maestro al estudiante de la etapa 1 y
   genera sus trayectorias con la rejilla de 15 puntos.
3. **Compuertas**: idénticas al ADR 0008 — la etapa 1 debe producir ≥ 6/8
   rostros válidos antes de pagar la etapa 2; el éxito de la iteración es
   ≥ 6/8 a 8 pasos con fidelidad comparable al maestro (±1 caso en la tasa de
   atributos).
4. Si esta reformulación también falla la compuerta de etapa 1, la evidencia
   dirá que el prior de 994 M no es destilable a 8 pasos con este presupuesto
   y las opciones se reducen a (c) o a una reducción arquitectónica profunda
   (ADR propio); no se pagan más iteraciones del mismo mecanismo.
5. El tamaño del artefacto (~1 GB en INT8 vs. RNF-01) sigue diferido al ADR de
   reducción arquitectónica, como en el ADR 0008.

## Consecuencias

- Diseño fijado en `docs/distill-trajectory-design.md` antes de pagar GPU;
  `scripts/distill_wuerstchen_prior.py` gana el modo `--mode trajectory`
  conservando `--mode marginal` para reproducibilidad histórica.
- El entrenamiento sin maestro abarata cada paso (1 forward + backward del
  estudiante, sin forwards guiados): la etapa 1 cabe holgadamente en una
  sesión de RTX 4090, incluida la generación de trayectorias.
- La destilación sobre trayectorias no mejora el techo de fidelidad del
  maestro (color de ojos, pecas, pendientes); lo conserva. Es el comportamiento
  esperado y no invalida la corrida.
- El resto del pipeline de integración no cambia: destilación → exportación
  ONNX → cuantización selectiva (ADR 0006) → benchmark en el dispositivo de
  referencia.
