#!/usr/bin/env bash
# Evaluación visual congelada de un checkpoint LoRA (conjunto de escala-1).
#
# Genera los 8 prompts fijados en docs/lora-scale-1-design.md con el
# checkpoint indicado y con --base-only (misma seed 42), usando la receta
# oficial: 1024 px, 30 timesteps, guía 8.0 y prompt negativo con términos de
# menores. Sirve igual para escala-2 y posteriores: las comparaciones entre
# corridas sólo son válidas con este mismo conjunto.
#
# Uso en la instancia, desde la raíz del repo:
#   bash scripts/eval-visual-lora.sh CHECKPOINT.pt DIRECTORIO_SALIDA \
#     [ARGS_EXTRA_VALIDADOR] [--skip-base]
# Ejemplo (línea base del ADR 0008, sin repetir los base-only de escala-3):
#   bash scripts/eval-visual-lora.sh ckpt.pt out "--prior-timesteps 12" --skip-base
set -euo pipefail

checkpoint=${1:?Uso: eval-visual-lora.sh CHECKPOINT.pt DIRECTORIO_SALIDA [ARGS_EXTRA] [--skip-base]}
output=${2:?Uso: eval-visual-lora.sh CHECKPOINT.pt DIRECTORIO_SALIDA [ARGS_EXTRA] [--skip-base]}
extra=${3:-}
skip_base=${4:-}
root=$(pwd)
mkdir -p "$output"

prompts=(
"flat vector avatar face of an adult, happy expression, square face, porcelain skin tone, side-parted black hair, green eyes with earrings, sky background"
"flat vector avatar face of an adult, confident expression, heart face, light skin tone, bob brown hair, gray eyes with freckles, lavender background"
"flat vector avatar face of an adult, calm expression, oval face, deep skin tone, curly pink hair, blue eyes with round glasses, sky background"
"flat vector avatar face of an adult, smiling expression, round face, brown skin tone, short blue hair, brown eyes without accessories, mint background"
"flat vector avatar face of an adult, happy expression, heart face, golden skin tone, bob blonde hair, green eyes without accessories, coral background"
"flat vector avatar face of an adult, calm expression, square face, tan skin tone, curly auburn hair, gray eyes with earrings, sand background"
"flat vector avatar face of an adult, confident expression, oval face, brown skin tone, short black hair, brown eyes without accessories, lavender background"
"flat vector avatar face of an adult, smiling expression, round face, porcelain skin tone, side-parted pink hair, blue eyes with freckles, mint background"
)

for i in "${!prompts[@]}"; do
  n=$(printf "%02d" $((i + 1)))
  if [[ "$skip_base" != "--skip-base" ]]; then
    echo "=== base-only $n ==="
    python scripts/validate_wuerstchen_lora.py \
      --root "$root" --base-only \
      --prompt "${prompts[$i]}" \
      --output "$output/base-$n.png"
  fi
  echo "=== lora $n ==="
  # shellcheck disable=SC2086
  python scripts/validate_wuerstchen_lora.py \
    --root "$root" --checkpoint "$checkpoint" \
    --prompt "${prompts[$i]}" $extra \
    --output "$output/lora-$n.png"
done
echo "EVAL_VISUAL_COMPLETA: $output"
