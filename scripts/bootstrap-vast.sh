#!/usr/bin/env bash
# Prepara una instancia Vast NUEVA para el entrenamiento LoRA (escala-1/2).
#
# Requisitos previos (ruta directa, ambos < 100 MB), desde la máquina local:
#   scp -P PUERTO transfer/avatarface-training-procedural-v2-dataset.tar \
#       transfer/avatarface-smoke-dataset.tar transfer/SHA256SUMS \
#       root@IP:/tmp/avatarface-transfer/
#
# Uso en la instancia:  bash scripts/bootstrap-vast.sh [/workspace/AvatarFace]
#
# Idempotente: reclona o actualiza el repo, instala deps fijadas del piloto,
# restaura los datasets verificando SHA-256, descarga los pesos públicos con
# verificación completa y ejecuta el preflight CUDA. No entrena nada.
set -euo pipefail

repo=${1:-/workspace/AvatarFace}
transfer_dir=${2:-/tmp/avatarface-transfer}

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
  "$transfer_dir/avatarface-training-procedural-v2-dataset.tar" \
  "$transfer_dir/SHA256SUMS" "$repo"
scripts/restore-from-drive.sh \
  "$transfer_dir/avatarface-smoke-dataset.tar" \
  "$transfer_dir/SHA256SUMS" "$repo"

mkdir -p models/wuerstchen-v2
cp transfer/model-manifest-trimmed-20260815.json models/wuerstchen-v2/model-manifest.json
python scripts/download-wuerstchen-weights.py \
  --manifest models/wuerstchen-v2/model-manifest.json

avatar-face verify-frozen-dataset \
  --manifest data/training-procedural-v2/manifest.json \
  --lock data/training-procedural-v2/dataset-v2.0.0.lock.json
avatar-face audit-dataset --manifest data/training-procedural-v2/manifest.json
avatar-face preflight-vast
echo "BOOTSTRAP_OK: instancia lista para entrenar"
