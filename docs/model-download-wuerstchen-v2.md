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
- SHA-256 del manifiesto:
  `e8e7ce4f5969f557fb1d078cbaa55c85a1ac943ceb079c0ca2198fc1ed770b38`.
- Contenido declarado: 3 componentes, 33 archivos y 30,005,997,031 bytes.
- Paquete: `transfer/avatarface-wuerstchen-v2.tar`.
- SHA-256 del paquete:
  `691b3775232c511ec6fbb1cdbed1060dc4ff1fa50b70816c7f80f2250739dead`.

El paquete excede 100 MB, por lo que sigue la ruta obligatoria **local → Google
Drive → Vast.ai**. Antes de extraer en Vast se valida `SHA256SUMS`; tras extraer
se vuelve a calcular cada hash según `model-manifest.json`. Los pesos, paquetes
de transferencia y manifiestos generados están ignorados por Git.
