#!/usr/bin/env bash
# Prepara una instancia Vast NUEVA para el entrenamiento LoRA (escala sobre v2.x).
#
# Requisitos previos (ruta directa, ambos < 200 MB), desde la máquina local:
#   scp -P PUERTO transfer/avatarface-training-procedural-v2-2.tar \
#       transfer/avatarface-smoke-procedural.tar transfer/SHA256SUMS \
#       root@IP:/tmp/avatarface-transfer/
#
# Uso en la instancia:  bash scripts/bootstrap-vast.sh [repo] [transfer_dir] \
#     [dataset_dir] [dataset_lock]
# Por defecto restaura data/training-procedural-v2-1 con su lock v2.1.0;
# para la release 512 px:  bash scripts/bootstrap-vast.sh /workspace/AvatarFace \
#     /tmp/avatarface-transfer training-procedural-v2-2 dataset-2.2.0.lock.json
#
# Idempotente: reclona o actualiza el repo, instala deps fijadas del piloto,
# restaura los datasets verificando SHA-256, descarga los pesos públicos con
# verificación completa y ejecuta el preflight CUDA. No entrena nada.
set -euo pipefail

repo=${1:-/workspace/AvatarFace}
transfer_dir=${2:-/tmp/avatarface-transfer}
dataset_dir=${3:-training-procedural-v2-1}
dataset_lock=${4:-dataset-v2.1.0.lock.json}

if [[ ! -d "$repo/.git" ]]; then
  git clone --depth 1 https://github.com/dasilvabalautaro/Avatar.git "$repo"
else
  git -C "$repo" pull --ff-only
fi
cd "$repo"

# Entorno de la imagen Vast (ver /etc/vast-agents-guide.md).
source /venv/main/bin/activate
pip install --no-cache-dir \
  "diffusers==0.39.0" "transformers==5.15.0" "peft==0.20.0" \
  "accelerate==1.14.0" "safetensors==0.8.0"
pip install --no-cache-dir -e .

scripts/restore-from-drive.sh \
  "$transfer_dir/avatarface-${dataset_dir}.tar" \
  "$transfer_dir/SHA256SUMS" "$repo"
scripts/restore-from-drive.sh \
  "$transfer_dir/avatarface-smoke-procedural.tar" \
  "$transfer_dir/SHA256SUMS" "$repo"

# El preflight exige el SHA256SUMS dentro del repo; copiarlo junto al manifiesto.
cp "$transfer_dir/SHA256SUMS" transfer/SHA256SUMS
mkdir -p models/wuerstchen-v2
cp transfer/model-manifest-trimmed-20260815.json models/wuerstchen-v2/model-manifest.json
python scripts/download-wuerstchen-weights.py \
  --manifest models/wuerstchen-v2/model-manifest.json

avatar-face verify-frozen-dataset \
  --manifest "data/${dataset_dir}/manifest.json" \
  --lock "data/${dataset_dir}/${dataset_lock}"
avatar-face audit-dataset --manifest "data/${dataset_dir}/manifest.json"
avatar-face preflight-vast
echo "BOOTSTRAP_OK: instancia lista para entrenar"
