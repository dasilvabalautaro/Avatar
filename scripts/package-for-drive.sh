#!/usr/bin/env bash
# Empaqueta el dataset declarado, sin descargar ni iniciar entrenamiento.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
target=${1:-"$root/data/smoke-procedural"}
output_dir=${2:-"$root/transfer"}
target=$(cd "$target" && pwd)
case "$target" in
  "$root"/data/*|"$root"/models/*) target_relative=${target#"$root"/} ;;
  *) echo "ERROR: el objetivo debe estar bajo $root/data o $root/models" >&2; exit 2 ;;
esac
target_name=$(basename "$target")
package="$output_dir/avatarface-${target_name}.tar"
sums="$output_dir/SHA256SUMS"

if [[ ! -f "$target/manifest.json" && ! -f "$target/model-manifest.json" ]]; then
  echo "ERROR: falta manifiesto en $target" >&2
  exit 2
fi
mkdir -p "$output_dir"
[[ ! -e "$package" ]] || {
  echo "ERROR: el paquete ya existe; una transferencia no se sobrescribe" >&2
  exit 3
}

# La ruta relativa fija permite restaurar sólo bajo el workspace remoto.
COPYFILE_DISABLE=1 tar -C "$root" --exclude='._*' -cf "$package" "$target_relative"
tar -tf "$package" >/dev/null
(cd "$output_dir" && shasum -a 256 "$(basename "$package")") >> "$sums"

package_bytes=$(stat -f '%z' "$package" 2>/dev/null || stat -c '%s' "$package")
if (( package_bytes > 100000000 )); then
  route="Drive → Vast.ai (supera 100 MB)"
else
  route="máquina local → Vast.ai (hasta 100 MB)"
fi

echo "Paquete creado: $package"
echo "Hashes: $sums"
echo "Tamaño: $package_bytes bytes; ruta autorizada: $route"
