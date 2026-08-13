#!/usr/bin/env bash
# Empaqueta el dataset declarado, sin descargar ni iniciar entrenamiento.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dataset=${1:-"$root/data/smoke-procedural"}
output_dir=${2:-"$root/transfer"}
package="$output_dir/avatarface-smoke-dataset.tar"
sums="$output_dir/SHA256SUMS"

if [[ ! -f "$dataset/manifest.json" ]]; then
  echo "ERROR: falta $dataset/manifest.json" >&2
  exit 2
fi
mkdir -p "$output_dir"
rm -f "$package" "$sums"

# La ruta relativa fija permite restaurar sólo bajo el workspace remoto.
if [[ "$dataset" != "$root/data/smoke-procedural" ]]; then
  echo "ERROR: este empaquetador sólo admite $root/data/smoke-procedural" >&2
  exit 2
fi
COPYFILE_DISABLE=1 tar -C "$root" --exclude='._*' -cf "$package" data/smoke-procedural
tar -tf "$package" >/dev/null
(cd "$output_dir" && shasum -a 256 "$(basename "$package")") > "$sums"

echo "Paquete creado: $package"
echo "Hashes: $sums"
