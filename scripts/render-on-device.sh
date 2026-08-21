#!/usr/bin/env bash
# Dibuja la galería con el trazado nativo de Android y la compara con la de
# Python (ADR 0012). Instala el APK, ejecuta el modo `render`, baja los PNG y
# mide la diferencia entre ambos trazados.
#
# El serial es OBLIGATORIO: nunca se ejecuta ADB sin -s.
#
# Uso:  scripts/render-on-device.sh <serial> [runs]
set -euo pipefail

serial=${1:?Falta el serial del dispositivo (adb devices)}
runs=${2:-5}
package=com.avatarface.app
adb=${ADB:-adb}
root=$(cd "$(dirname "$0")/.." && pwd)
output="$root/artifacts/render-android"

cd "$root"

# Las especificaciones viajan como asset para que la app y Python dibujen
# exactamente las mismas personas.
python scripts/render_gallery.py --dump-specs android/app/src/main/assets/gallery-specs.json
python scripts/dump_parser_cases.py --output android/app/src/main/assets/parser-cases.json

( cd android && ./gradlew --no-daemon :app:assembleDebug )

"$adb" -s "$serial" install -r android/app/build/outputs/apk/debug/app-debug.apk
"$adb" -s "$serial" shell am force-stop "$package"
"$adb" -s "$serial" shell run-as "$package" rm -rf files/render files/benchmark-result.json
"$adb" -s "$serial" shell am start -W -n "$package/.MainActivity" \
  --es mode render --ei runs "$runs" >/dev/null

# La actividad escribe el resultado al terminar; se espera a que aparezca.
for _ in $(seq 1 60); do
  if "$adb" -s "$serial" shell run-as "$package" \
      test -f files/benchmark-result.json 2>/dev/null; then
    break
  fi
  sleep 1
done

rm -rf "$output"
mkdir -p "$output"
"$adb" -s "$serial" shell run-as "$package" cat files/benchmark-result.json \
  > "$output/render-result.json"
for name in $("$adb" -s "$serial" shell run-as "$package" ls files/render); do
  name=$(printf '%s' "$name" | tr -d '\r')
  "$adb" -s "$serial" shell run-as "$package" cat "files/render/$name" > "$output/$name"
done
"$adb" -s "$serial" shell am force-stop "$package"

python - "$output/render-result.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print(f"dispositivo: {data['avatars']} avatares de {data['image_size']} px, "
      f"mediana {data['median_ms']:.1f} ms, máximo {data['max_ms']:.1f} ms")
# El parser de texto también existe en los dos lenguajes: sus casos vienen de
# Python como asset y la app informa de cualquier interpretación distinta.
mismatches = data.get("parser_mismatches", [])
if mismatches:
    for item in mismatches[:10]:
        print(f"  parser difiere: {item['attribute']!r} en {item['text']!r}: "
              f"python={item['python']!r} android={item['android']!r}")
    raise SystemExit(f"parser_fallido: {len(mismatches)} discrepancias")
print("parser_ok: las dos implementaciones interpretan igual las mismas frases")
PY

python scripts/compare_android_render.py --android "$output" --python artifacts/render-demo
