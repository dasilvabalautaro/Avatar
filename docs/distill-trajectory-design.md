# Diseño de la destilación por trayectorias reales — 2026-08-18

Diseño de la reformulación de la destilación del prior Würstchen Stage C según
el ADR 0009, tras los tres fallos de compuerta de la destilación sobre la
marginal N(0, I) (etapas 1, 1b y 1c, 2026-08-18). La configuración queda
fijada **antes** de pagar cualquier corrida. Sustituye al mecanismo de
`docs/distill-prior-design.md`; conserva sus compuertas.

## Punto de partida

- Maestro de la etapa 1: prior base + LoRA de escala-3 (500 pasos, `lr=5e-5`,
  256 px, rango 16; SHA-256 `fbc61da9...`), receta oficial: rejilla de 30
  timesteps, guía 8.0. Maestro de la etapa 2: el estudiante de la etapa 1.
- El prior opera sobre latentes EffNet de 16 canales: a 256 px son mapas de
  8×8. La destilación sigue sin necesitar imágenes, sólo captions.
- Dataset: los 818 captions del split train de la release v2.1.0 (congelada,
  hash verificado).
- Diagnóstico que motiva el cambio (etapa 1c): el estudiante aprende rasgos
  completos (señal correcta) pero los repite en mosaico — entrena sobre ruido
  puro N(0, I) y en inferencia recibe estados intermedios reales que nunca
  vio. La reformulación cambia **sólo** la distribución de entrada; todo lo
  demás se conserva para atribución limpia.

## Mecanismo (dos fases por etapa)

**Fase 1 — generación de trayectorias** (una sola vez por etapa):

1. Para cada caption de train, en orden determinista con semilla 42, se
   muestrea ruido inicial x ~ N(0, I) y el maestro ejecuta su trayectoria
   guiada completa (guía 8.0 ya combinada: el estudiante aprende la salida
   guiada y en el dispositivo no se duplica el coste por paso).
2. Se almacenan los latentes en todos los puntos de la rejilla del maestro
   (etapa 1: 30 puntos; etapa 2: 15). Volumen: 818 × 30 × (16×8×8) en bf16
   ≈ 50 MB — cabe en VRAM.
3. También se cachean los embeddings de texto de los 818 captions (~126 MB en
   bf16): el entrenamiento no vuelve a codificar.
4. El maestro se libera de VRAM al terminar la fase.

**Fase 2 — entrenamiento** (sin maestro):

1. Se muestrea un caption (ciclado en orden determinista) y un salto k de la
   rejilla del estudiante (etapa 1: 14 saltos, del punto 2k al 2k+2 del
   maestro).
2. Entrada del estudiante: el latente real x(2k) de la trayectoria; objetivo:
   el epsilon efectivo del salto despejado de los latentes reales x(2k) y
   x(2k+2) con los mismos coeficientes del scheduler ya usados en 1-1c.
3. Pérdida: MSE normalizada por la potencia media del epsilon objetivo
   (mse/‖eps‖², la formulación de la etapa 1c que restauró la señal).
4. Cada paso cuesta 1 forward + 1 backward del estudiante: mucho más barato
   que en 1-1c (4 forwards del maestro + 1 del estudiante).

## Configuración fijada

| Parámetro | Valor | Justificación |
|---|---|---|
| Etapas | 2: 30→15 y 15→8 pasos | Igual que el ADR 0008 |
| Modo | `--mode trajectory` | ADR 0009; la marginal queda descartada |
| Resolución de destilación | 256 (latentes 8×8) | Resolución de producto (ADR 0007) |
| Captions | 818 de train v2.1.0, ciclados en orden determinista | Igual que 1-1c |
| Pasos de entrenamiento | 6000 por etapa | Igual que 1c (única iteración con señal correcta) |
| Learning rate | 5e-5, AdamW, estudiante fp32 | Igual que 1c |
| Pérdida | MSE/‖eps‖² (`--normalize-target`) | Igual que 1c |
| Semilla | 42 | Regla del proyecto |
| Estudiante | copia completa del prior (994 M) con LoRA escala-3 fusionado | Igual que 1-1c |
| VRAM estimada | ~15 GB en fase 1 (maestro bf16 + estudiante + AdamW); ~13 GB en fase 2 (sin maestro) | Cabe en la RTX 4090 de 48 GiB |
| Tiempo estimado por etapa | < 1 h (fase 1: 818 × 30 pasos guiados ≈ 15-25 min; fase 2: 6000 pasos baratos) | Verificar con el log; límite de coste ≤ 2 h por etapa |

Script: el mismo `scripts/distill_wuerstchen_prior.py` con `--mode trajectory`
(el modo `--mode marginal` se conserva para reproducibilidad histórica). La
salida sigue siendo `pilot-checkpoint.pt` con `distill_config` embebida
(ahora con el campo `mode`). Validación: `scripts/validate_wuerstchen_lora.py`
con `--distilled-checkpoint` y `--prior-timesteps 15` (etapa 1) u `8` (etapa
2), sin cambios.

## Conjunto de evaluación y compuertas

Idénticas al ADR 0008 y al diseño anterior:

- Los **ocho prompts fijos** del conjunto congelado
  (`docs/lora-scale-1-design.md`), con la lista de verificación vigente:
  rostro válido (un único avatar, sin mosaico), atributos, edad aparente
  adulta (RF-09), divergencia vs. base-only. Comparación contra el maestro a
  30 pasos (escala-3, ya existente) y contra las etapas 1b/1c al mismo número
  de pasos (ya existentes).
- **Compuerta de etapa 2**: la etapa 1 (15 pasos) debe producir ≥ 6/8 rostros
  válidos; si no, no se paga la etapa 2 y se documenta.
- **Compuerta de éxito de la iteración**: el estudiante a 8 pasos debe
  producir ≥ 6/8 rostros válidos y fidelidad comparable al maestro (±1 caso
  en la tasa de atributos). Entonces sigue exportación ONNX y cuantización
  selectiva (ADR 0006) y benchmark en el TECNO KM5s.
- Si la etapa 1 falla de nuevo, no se pagan más iteraciones del mecanismo:
  conforme al ADR 0009, las opciones pasan a ser (c) o reducción
  arquitectónica.
- Respaldo local con SHA-256 de checkpoints y muestras; documento de
  resultados `docs/experiments/wuerstchen-distill-stage-1d-AAAA-MM-DD.md`;
  instancia destruida al terminar (GPU al 0 %).

## Riesgos conocidos

- Los estados del estudiante en inferencia se desvían de los del maestro a
  medida que avanza la cadena (error compuesto): la trayectoria del maestro es
  una aproximación, no la distribución exacta del estudiante. Si el defecto
  residual es leve, se evalúa una segunda etapa de corrección; si es grave,
  aplica la salida del ADR 0009.
- El mosaico de 1c podría tener también un componente de capacidad (lr alto
  sobre 14 saltos). Si la etapa 1d muestra rasgos completos sin mosaico pero
  con deriva de atributos, el ajuste siguiente es de lr/pasos, no de método.
- El estudiante hereda el techo de fidelidad del maestro (color de ojos,
  pecas, pendientes): comportamiento esperado, no invalida la corrida.
- El tamaño del artefacto (~1 GB en INT8) puede superar RNF-01 (250 MB):
  problema del ADR futuro de reducción arquitectónica, no de esta iteración.
