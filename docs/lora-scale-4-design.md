# Diseño del experimento LoRA escala-4 — 2026-08-17

Diseño del siguiente entrenamiento LoRA sobre Würstchen v2 Stage C, posterior a
escala-3 (`docs/experiments/wuerstchen-lora-scale-3-2026-08-17.md`). Responde a
la decisión del usuario tras escala-3: probar en una sola corrida las dos
palancas de capacidad restantes —**resolución de entrenamiento y
rango/módulos del LoRA**— antes de aceptar el techo del piloto y pasar a
integración con destilación (ADR 0007). La configuración queda fijada **antes**
de pagar cualquier corrida.

## Punto de partida

- Escala-2 demostró que más pasos (500, `lr=5e-5`) no compran fidelidad en
  color de ojos, pecas ni pendientes; escala-3 demostró que tampoco la compra
  un dataset con el doble de muestras y captions más finos (v2.1.0).
- Hallazgo estructural nuevo (2026-08-17): las imágenes del dataset son
  nativas de 256 px y el encoder EffNetV2-S reduce 32×, así que el prior
  entrena sobre latentes de sólo 8×8 celdas. Un iris (12 px) ocupa ~0.4 celdas
  y una peca (4 px) ~0.1: los atributos difíciles literalmente no sobreviven
  al encoder a 256 px.
- El generador procedimental escalaba el dibujo en coordenadas absolutas;
  ahora escala con `image_size` y su salida a 256 px quedó verificada
  bit-idéntica (hashes de manifiesto v1.1.0 y v2.1.0 sin cambio).
- Nuevas palancas ya implementadas: `--resolution`, `--lora-rank`,
  `--lora-alpha`, `--lora-dropout` y `--lora-modules` en
  `scripts/run_wuerstchen_lora_pilot.py` (la config viaja en el checkpoint y
  el validador la respeta, con respaldo a los valores históricos).

## Hipótesis a discriminar

1. **Resolución**: a 512 px el prior entrena sobre latentes de 16×16; el iris
   pasa a ~24 px (~0.75 celdas) y los pendientes a ~20×28 px (~1 celda). Si el
   límite era la señal que sobrevive al encoder, la fidelidad de ojos y
   pendientes mejora.
2. **Capacidad del adaptador**: rango 16 sólo en atención puede no tener
   espacio para todas las correspondencias caption → rasgo fino. Rango 32 y
   módulos FFN adicionales amplían esa capacidad sin tocar el dataset.
3. Si **ninguna** de las dos mejora, la evidencia de que el techo está en el
   modelo base queda sólida y se pasa a integración con destilación (ADR 0007).

Las pecas (~8 px a 512, ~0.25 celdas) pueden seguir bajo el umbral del
encoder; si todo lo demás mejora y las pecas no, la palanca siguiente sería
reforzar su tamaño en el generador, no otra corrida ciega.

## Configuración de escala-4

| Parámetro | Valor | Justificación |
|---|---|---|
| Dataset | release v2.2.0, 1024 muestras nativas de 512 px | lock `dataset-2.2.0.lock.json`, manifiesto SHA-256 `53fa3b374dc1ed48e59a4510b2348fd43ef6c12027420ccc0fa69e4f4a2616f8` |
| Pasos | 500 | Igual que escala-2/3 para comparabilidad |
| Learning rate | 5e-5 | Valor estable verificado en escala 1-3 |
| Resolución de entrenamiento | 512 | Hipótesis 1; exige dataset nativo 512 (v2.2.0) |
| LoRA | rango 32, alpha 32, dropout 0.05 | Hipótesis 2; alpha = rango mantiene la escala efectiva |
| Módulos LoRA | `to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2` | Hipótesis 2; el script aborta si no engancha ningún módulo |
| Batch | 1 | Sin cambios de infraestructura |
| Semilla | 42 | Comparabilidad con escala 1-3 y `base-only` |
| Precisión | bf16 | Igual que el piloto |
| Límite de coste | ≤ 2 horas de GPU | 512 px ≈ 2-3× el coste por paso de escala-3 |

Comando de entrenamiento en la instancia:

```bash
python scripts/run_wuerstchen_lora_pilot.py \
  --root /workspace/AvatarFace \
  --dataset-dir data/training-procedural-v2-2 \
  --steps 500 --learning-rate 5e-5 \
  --resolution 512 \
  --lora-rank 32 --lora-alpha 32 \
  --lora-modules to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2 \
  --output artifacts/lora-scale-4 2>&1 | tee train-scale-4.log
```

Si la muestra de escala-4 resulta inválida (ruido, mosaico, blanco, embeddings
no finitos), se repite una única vez volviendo a rango 16 y sólo atención, pero
manteniendo 512 px: así se aisla si el problema lo introdujo la capacidad
extra o la resolución. Si la fidelidad de atributos difíciles no mejora
respecto a escala-3, no se paga ninguna corrida más de este tipo: se acepta
el techo y se pasa a la fase de integración (ADR 0007).

## Conjunto fijo de evaluación visual

Los mismos ocho prompts de escala-1/2/3 (todos de adultos, RF-09), con la
receta oficial: 30 timesteps, fp16, 1024 px, guía 8.0, prompt negativo con
términos de menores, seed 42. Cada prompt se evalúa también con `--base-only`.
La comparación directa es contra escala-3 (mismo dataset base de captions, sólo
cambia resolución y LoRA).

### Lista de verificación por muestra

1. Rostro válido: sin ruido, mosaico ni imagen en blanco.
2. Atributos del prompt presentes, con foco en los que fallaron en escala 1-3:
   color de ojos, pecas, pendientes, rubio del caso 5.
3. **Edad aparente adulta** (RF-09): cualquier muestra dudosa invalida la
   corrida.
4. Divergencia respecto a `base-only` documentada (idéntica, sutil, clara).

## Compuertas y respaldo

- Instancia nueva vía `scripts/bootstrap-vast.sh` con
  `training-procedural-v2-2` y `dataset-2.2.0.lock.json`; la instancia de
  escala-3 quedó encendida pero se descarta su reutilización para aislar la
  corrida (el usuario decide apagarla al contratar la nueva).
- Respaldo local con SHA-256 de `pilot-checkpoint.pt` y de las 16 imágenes en
  `artifacts/lora-scale-4/`; los hashes se registran en el documento de
  resultados.
- Al terminar: detener la instancia Vast y verificar 0 % de utilización.
- El documento de resultados se creará como
  `docs/experiments/wuerstchen-lora-scale-4-AAAA-MM-DD.md` con pérdidas por
  paso, parámetros entrenables, hashes y la evaluación visual completa.
