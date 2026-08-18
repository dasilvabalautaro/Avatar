# Destilación del prior, etapa 1 (30→15 pasos) — 2026-08-18

Primera iteración de la destilación progresiva diseñada en
`docs/distill-prior-design.md` (ADR 0008). Resultado: **compuerta no
superada**; la etapa 2 no se paga, conforme al diseño.

## Configuración ejecutada

| Parámetro | Valor |
|---|---|
| Maestro | prior base + LoRA escala-3 (SHA-256 `fbc61da9...`), guía 8.0, rejilla oficial de 30 |
| Estudiante | prior completo (994 M) con LoRA fusionado, fp32, AdamW `lr=1e-5` |
| Rejilla del estudiante | 15 puntos (pares de la oficial), 14 saltos |
| Pasos | 2000, captions de train v2.1.0, semilla 42 |
| Checkpoint | `artifacts/distill-stage-1/pilot-checkpoint.pt`, SHA-256 `888639324318b6ceb3ba014badc2dc8c32a4930195aeba857c98194f51d9798b` |

## Trayectoria de pérdida (referencia)

paso 1 = 3.608661, 100 = 0.718738, 500 = 0.000004, 1000 = 0.438071,
1500 = 0.035884, 2000 = 1.599620. La dispersión entre pasos es de órdenes de
magnitud: la pérdida de cada paso depende del salto k muestreado.

## Evaluación visual (15 timesteps, 8 prompts fijos)

Las 8 muestras pasaron `validation_ok` (embeddings finitos, pixel_std en
rango), pero **ninguna contiene un rostro**: campos de color planos o manchas
difusas sin estructura facial. **0/8 válidas; compuerta de etapa 2 (≥ 6/8)
no superada.**

## Diagnóstico

El mecanismo falló en la ponderación de la pérdida, no en la tubería:

1. El peso SNR implementado, `w(t) = ab(t)/(1-ab(t))` normalizado a media 1,
   asigna peso casi nulo a los saltos de **alto ruido** (t cercano a 1, donde
   `ab(t)` tiende al mínimo de 1e-4). Esos saltos son precisamente donde se
   forma la estructura global (la cara): el estudiante casi no recibió
   gradiente para ellos y en evaluación parte de ruido puro sin saber
   estructurarlo — de ahí los campos de color.
2. La pérdida casi nula del paso 500 (0.000004) confirma que algunos saltos
   quedaron prácticamente sin señal efectiva.
3. La pérdida final alta (1.60) en otros saltos indica además dispersión de
   magnitudes del epsilon objetivo entre saltos que la normalización no
   resolvió.

## Siguiente paso propuesto (decisión del usuario)

Repetir **sólo la etapa 1** con la corrección del mecanismo, como preveía el
diseño: pérdida con **peso uniforme** (sin SNR) y resto idéntico (mismo
maestro, 2000 pasos, `lr=1e-5`, seed 42). Si la segunda etapa 1 supera la
compuerta (≥ 6/8), se continúa con la etapa 2 (15→8) según el diseño; si no,
se revisa la formulación del objetivo (coeficientes del salto) antes de pagar
más GPU.

## Hashes (SHA-256)

Los 8 PNG de evaluación, los dos logs y el checkpoint están listados en
`artifacts/distill-stage-1/SHA256SUMS`; la transferencia del checkpoint se
hizo por bloques con verificación final contra ese manifiesto. La GPU quedó a
`0 %` de utilización al cerrar cada sesión; la instancia sigue encendida a la
espera de la decisión del usuario.
