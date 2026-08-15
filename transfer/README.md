# Transferencias: local → Drive → Vast.ai

Este directorio contiene las herramientas para transferir artefactos sin
credenciales. Los paquetes `.tar` y `SHA256SUMS` son locales e ignorados por Git;
los enlaces/IDs de Drive y cualquier secreto no se escriben aquí.

## Regla de transporte

El umbral es **100 MB = 100,000,000 bytes** del archivo `.tar` final:

- Si el `.tar` supera 100 MB, se sube a Google Drive y Vast.ai lo descarga desde
  allí.
- Si mide 100 MB o menos, se transfiere directamente desde esta máquina local a
  Vast.ai.

Siempre se transfiere también el `SHA256SUMS` correspondiente y se verifica el
hash antes de extraer. Para pesos, se incluye `model-manifest.json` con la
revisión, licencia, tamaño y SHA-256 de cada archivo. El empaquetador informa la ruta autorizada; no sube ni
transfiere automáticamente.

## Antes de crear una instancia

Desde la raíz del repositorio, auditar los datos y preparar un paquete:

```bash
.venv/bin/avatar-face audit-dataset --manifest data/training-procedural-v2/manifest.json
.venv/bin/avatar-face verify-frozen-dataset \
  --manifest data/training-procedural-v2/manifest.json \
  --lock data/training-procedural-v2/dataset-v2.0.0.lock.json
scripts/package-for-drive.sh data/training-procedural-v2
.venv/bin/avatar-face preflight-vast --local \
  --manifest data/training-procedural-v2/manifest.json
```

El último comando sólo prueba el paquete en macOS/Linux local: no puede validar
CUDA. El paquete actual mide 2,577,920 bytes (≈2.5 MB), así que sigue la ruta
directa máquina local → Vast.ai; no se sube a Drive. Conservar el `.tar` y
`SHA256SUMS` hasta comprobar la restauración remota.

## En Vast.ai (sin entrenar todavía)

1. Cree una instancia con al menos 16 GiB de VRAM y 50 GiB libres. Use una imagen
   CUDA compatible con PyTorch 2.2.2.
2. Según el umbral: transfiera ambos archivos directamente desde la máquina
   local, o descárguelos desde Drive con reanudación a un directorio temporal
   (por ejemplo `/tmp/avatarface-transfer`). No añada URLs de Drive a Git ni al
   historial del shell.
3. Desde un checkout del repositorio, ejecute. El restaurador acepta exactamente
   un objetivo bajo `data/` o `models/`, borra el `.tar` remoto después de
   comprobar su SHA-256 y deja el manifiesto para la auditoría posterior.

```bash
scripts/restore-from-drive.sh /tmp/avatarface-transfer/avatarface-training-procedural-v2-dataset.tar \
  /tmp/avatarface-transfer/SHA256SUMS /workspace/AvatarFace
/workspace/AvatarFace/.venv/bin/avatar-face preflight-vast
```

La restauración verifica el listado y SHA-256, rechaza rutas inseguras del tar,
extrae únicamente bajo el destino y borra el tar temporal sólo después de
verificarlo. Para pesos, ejecute además
`python scripts/verify-model-manifest.py models/wuerstchen-v2/model-manifest.json`.
`preflight-vast` ejecuta una operación CUDA real, pero no entrena.

## Paquete fresco tras cuota de Drive (2026-08-15)

Si Drive mantiene la cuota al usar **Hacer una copia**, use el paquete generado
localmente `avatarface-wuerstchen-v2-stageb-fresh-20260815.tar` y su archivo
`SHA256SUMS-stageb-fresh-20260815`. Deben subirse como archivos nuevos; no deben
reemplazar ni renombrar los paquetes anteriores. El tar mide 34,309,578,752
bytes y su SHA-256 es
`c25196411187496755f8c1001a5ce90aa344aa32adabcf26c988d2a8d0a7a92a`.
No ejecute `train-smoke` ni una corrida pagada hasta que el reporte sea `ready`.
