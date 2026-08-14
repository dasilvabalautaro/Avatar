#!/usr/bin/env bash
# Comprobaciones de sólo lectura. No instala, descarga, empaqueta ni entrena.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="$root/.venv/bin/python"
[[ -x "$python_bin" ]] || python_bin=python
manifest="$root/data/smoke-procedural/manifest.json"
package_dir="$root/transfer"
min_vram=16
min_disk=50
local=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) manifest=$2; shift 2 ;;
    --package-dir) package_dir=$2; shift 2 ;;
    --min-vram-gib) min_vram=$2; shift 2 ;;
    --min-free-disk-gib) min_disk=$2; shift 2 ;;
    --local) local=true; shift ;;
    *) echo "ERROR: argumento desconocido: $1" >&2; exit 2 ;;
  esac
done

failures=()
need() { command -v "$1" >/dev/null || failures+=("falta_comando:$1"); }
need "$python_bin"
need tar
need shasum

if [[ ! -f "$manifest" ]]; then
  failures+=("falta_manifiesto:$manifest")
else
  if ! "$python_bin" - "$manifest" <<'PY'
import hashlib, json, sys
from pathlib import Path
path = Path(sys.argv[1]).resolve()
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload["samples"]
    assert payload.get("schema_version") in {1, 2} and isinstance(samples, list) and samples
    seen = set()
    for raw in samples:
        image = path.parent / raw["image"]
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        assert image.is_file() and digest == raw["sha256"] and digest not in seen
        seen.add(digest)
except (AssertionError, KeyError, OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"dataset/hash inválido: {error}")
print(f"dataset_samples={len(samples)} manifest_sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
PY
  then failures+=("dataset_o_hash_invalido"); fi
fi

if [[ -f "$package_dir/SHA256SUMS" ]]; then
  package_name="avatarface-$(basename "$(dirname "$manifest")")-dataset.tar"
  expected=$(awk -v name="$package_name" '$2 == name { print $1; exit }' "$package_dir/SHA256SUMS")
  if [[ -f "$package_dir/$package_name" ]]; then
    [[ -n "$expected" ]] || failures+=("paquete_no_listado:$package_name")
    actual=$(shasum -a 256 "$package_dir/$package_name" | awk '{print $1}')
    [[ "$actual" == "$expected" ]] || failures+=("hash_paquete_invalido:$package_name")
    tar -tf "$package_dir/$package_name" >/dev/null || failures+=("tar_invalido:$package_name")
  fi
else
  failures+=("falta_SHA256SUMS:$package_dir/SHA256SUMS")
fi

free_kib=$(df -Pk "$root" | awk 'NR==2 {print $4}')
free_gib=$(awk -v k="$free_kib" 'BEGIN { printf "%.2f", k / 1024 / 1024 }')
awk -v actual="$free_gib" -v required="$min_disk" 'BEGIN { exit !(actual >= required) }' || failures+=("disco_insuficiente:${free_gib}GiB<${min_disk}GiB")

if ! "$python_bin" - <<'PY'
import PIL, torch
print(f"python_ok torch={torch.__version__} pillow={PIL.__version__}")
PY
then failures+=("dependencias_python_invalidas"); fi

gpu="no comprobada"
if [[ "$local" == true ]]; then
  gpu="omitida_por_modo_local"
else
  if ! command -v nvidia-smi >/dev/null; then
    failures+=("falta_nvidia_smi")
  else
    gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tr '\n' ';')
    if ! "$python_bin" - "$min_vram" <<'PY'
import math, sys, torch
required = float(sys.argv[1]) * 1024**3
assert torch.cuda.is_available(), "CUDA no disponible para PyTorch"
assert torch.cuda.device_count() > 0, "sin GPU CUDA"
props = torch.cuda.get_device_properties(0)
assert props.total_memory >= required, f"VRAM insuficiente: {props.total_memory / 1024**3:.2f} GiB"
value = torch.randn((64, 64), device="cuda")
assert math.isfinite(float((value @ value).sum().cpu()))
print(f"cuda_ok gpu={torch.cuda.get_device_name(0)} vram_gib={props.total_memory / 1024**3:.2f}")
PY
    then failures+=("gpu_cuda_o_vram_invalida"); fi
  fi
fi

if (( ${#failures[@]} )); then
  printf '{"ready":false,"gpu":"%s","free_disk_gib":%s,"failures":[' "$gpu" "$free_gib"
  printf '"%s",' "${failures[@]}" | sed 's/,$//'
  echo ']}'
  exit 1
fi
printf '{"ready":true,"gpu":"%s","free_disk_gib":%s,"failures":[]}\n' "$gpu" "$free_gib"
