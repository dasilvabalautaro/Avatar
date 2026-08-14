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
expected=$(awk -v name="$package_name" '$2 == name { print $1; exit }' "$sums")
actual=$(shasum -a 256 "$package" | awk '{print $1}')
[[ "$actual" == "$expected" ]] || { echo "ERROR: SHA256 no coincide" >&2; exit 3; }

# Sólo un dataset relativo bajo data/; rechaza traversal y opciones de tar.
dataset_relative=""
while IFS= read -r member; do
  [[ "$member" == data/* ]] || {
    echo "ERROR: entrada inesperada: $member" >&2; exit 4;
  }
  [[ "$member" != *".."* && "$member" != /* ]] || {
    echo "ERROR: entrada insegura: $member" >&2; exit 4;
  }
  candidate=${member#data/}
  candidate=${candidate%%/*}
  [[ -n "$candidate" ]] || { echo "ERROR: raíz de dataset inválida" >&2; exit 4; }
  if [[ -z "$dataset_relative" ]]; then
    dataset_relative="data/$candidate"
  fi
  [[ "$member" == "$dataset_relative" || "$member" == "$dataset_relative"/* ]] || {
    echo "ERROR: el paquete contiene más de un dataset" >&2; exit 4;
  }
done < <(tar -tf "$package")

mkdir -p "$destination"
tar -C "$destination" -xf "$package"
[[ -n "$dataset_relative" && -f "$destination/$dataset_relative/manifest.json" ]] || {
  echo "ERROR: extracción incompleta" >&2; exit 5;
}
rm -f "$package"
echo "Restaurado y verificado en: $destination/$dataset_relative"
