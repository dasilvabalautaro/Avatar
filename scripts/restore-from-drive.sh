#!/usr/bin/env bash
# Restaura un paquete ya descargado de Drive. No recibe URLs ni credenciales.
set -euo pipefail

package=${1:?Uso: restore-from-drive.sh PAQUETE.tar SHA256SUMS DESTINO}
sums=${2:?Uso: restore-from-drive.sh PAQUETE.tar SHA256SUMS DESTINO}
destination=${3:?Uso: restore-from-drive.sh PAQUETE.tar SHA256SUMS DESTINO}

[[ -f "$package" ]] || { echo "ERROR: paquete inexistente" >&2; exit 2; }
[[ -f "$sums" ]] || { echo "ERROR: SHA256SUMS inexistente" >&2; exit 2; }
package_name=$(basename "$package")

grep -Fq "  $package_name" "$sums" || { echo "ERROR: paquete no listado" >&2; exit 3; }
(cd "$(dirname "$package")" && shasum -a 256 -c "$sums")

# Sólo rutas relativas bajo data/smoke-procedural; rechaza traversal y opciones de tar.
while IFS= read -r member; do
  [[ "$member" == data/smoke-procedural || "$member" == data/smoke-procedural/* ]] || {
    echo "ERROR: entrada inesperada: $member" >&2; exit 4;
  }
  [[ "$member" != *".."* && "$member" != /* ]] || {
    echo "ERROR: entrada insegura: $member" >&2; exit 4;
  }
done < <(tar -tf "$package")

mkdir -p "$destination"
tar -C "$destination" -xf "$package"
[[ -f "$destination/data/smoke-procedural/manifest.json" ]] || {
  echo "ERROR: extracción incompleta" >&2; exit 5;
}
rm -f "$package"
echo "Restaurado y verificado en: $destination/data/smoke-procedural"
