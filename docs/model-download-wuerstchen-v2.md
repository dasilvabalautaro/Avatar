# Evidencia de descarga — Würstchen v2

Fecha: 2026-08-13. Los pesos se descargaron exclusivamente en la máquina local;
no se descargaron desde Vast.ai.

## Componentes fijados

| Componente | Repositorio | Revisión | Licencia | Uso |
|---|---|---|---|---|
| Prior Stage C base | `warp-ai/wuerstchen-prior-model-base` | `3f9205c8c2e7cf103192954fe6f096e66f9d4efc` | MIT | Único componente que se adaptará con LoRA. |
| Decoder Stage A/B | `warp-ai/wuerstchen` | `c3da41406ddd4d9c48c49aa93981a82354351b83` | MIT | Congelado. |
| Encoder OpenCLIP ViT-bigG/14 | `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` | `743c27bd53dfe508a0ade0f50698f99b39d03bec` | MIT | Congelado. |

Se eliminaron las copias `.bin` duplicadas; la release conserva exclusivamente
`safetensors` y los archivos de configuración/tokenización requeridos.

## Integridad y transferencia

- Manifiesto local: `models/wuerstchen-v2/model-manifest.json`.
- SHA-256 del manifiesto (release con Stage B):
  `0b1c6b0ee3efdf4d1e5985e6728b962f9c996b7cd9f4bf71190c561959fe660a`.
- Contenido declarado: 4 componentes, 34 archivos y 34,309,480,546 bytes.
- Paquete Stage B: `transfer/avatarface-wuerstchen-v2-stageb.tar`.
- SHA-256 del paquete Stage B:
  `9568427b33d1dbc7eed68f331ab2b76173dc0e6bda79a12d3d1ce569ab8482be`.

El paquete excede 100 MB, por lo que sigue la ruta obligatoria **local → Google
Drive → Vast.ai**. Antes de extraer en Vast se valida `SHA256SUMS`; tras extraer
se vuelve a calcular cada hash según `model-manifest.json`. Los pesos, paquetes
de transferencia y manifiestos generados están ignorados por Git.

## Restauración remota auditada

El 2026-08-15 se restauró la copia Stage B descargada desde Drive en una RTX
4090 de 48 GB nominales. El SHA-256 del tar coincidió antes de extraer y la
auditoría remota confirmó los 34 archivos, 4 componentes y 34,309,480,546 bytes
con el SHA-256 `0b1c6b0ee3efdf4d1e5985e6728b962f9c996b7cd9f4bf71190c561959fe660a`.
El tar temporal se eliminó tras la extracción.

La prueba CUDA real pasó con PyTorch `2.12.0+cu130`: RTX 4090 con 47.37 GiB
visibles por PyTorch (la representación binaria habitual de una GPU comercial
de 48 GB). La instancia no posee volumen persistente; la copia local y Drive
siguen siendo los respaldos de autoridad.
