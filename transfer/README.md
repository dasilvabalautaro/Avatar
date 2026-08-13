# Transferencias: local → Drive → Vast.ai

Este directorio contiene las herramientas para transferir artefactos sin
credenciales. Los paquetes `.tar` y `SHA256SUMS` son locales e ignorados por Git;
los enlaces/IDs de Drive y cualquier secreto no se escriben aquí.

## Antes de crear una instancia

Desde la raíz del repositorio, auditar los datos y preparar un paquete:

```bash
.venv/bin/avatar-face audit-dataset --manifest data/smoke-procedural/manifest.json
scripts/package-for-drive.sh
.venv/bin/avatar-face preflight-vast --local
```

El último comando sólo prueba el paquete en macOS/Linux local: no puede validar
CUDA. Subir **sin modificar** `transfer/avatarface-smoke-dataset.tar` y
`transfer/SHA256SUMS` a una carpeta privada de Google Drive. Conservarlos hasta
que la restauración remota haya sido comprobada.

## En Vast.ai (sin entrenar todavía)

1. Cree una instancia con al menos 16 GiB de VRAM y 50 GiB libres. Use una imagen
   CUDA compatible con PyTorch 2.2.2.
2. Descargue ambos archivos desde Drive, con reanudación, a un directorio temporal
   (por ejemplo `/tmp/avatarface-transfer`). No añada la URL de Drive a Git ni al
   historial del shell.
3. Desde un checkout del repositorio, ejecute:

```bash
scripts/restore-from-drive.sh /tmp/avatarface-transfer/avatarface-smoke-dataset.tar \
  /tmp/avatarface-transfer/SHA256SUMS /workspace/AvatarFace
/workspace/AvatarFace/.venv/bin/avatar-face preflight-vast
```

La restauración verifica el listado y SHA-256, rechaza rutas inseguras del tar,
extrae únicamente bajo el destino y borra el tar temporal sólo después de
verificarlo. `preflight-vast` ejecuta una operación CUDA real, pero no entrena.
No ejecute `train-smoke` ni una corrida pagada hasta que el reporte sea `ready`.
