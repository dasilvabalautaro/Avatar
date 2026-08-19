# Dataset de destilación del maestro — etapa E2 del ADR 0010 (2026-08-19)

## Objetivo

Generar la release `avatarface-distill-teacher` v1.0.0: pares caption → imagen
del maestro (prior base + LoRA escala-3, receta oficial) para entrenar al
estudiante compacto (`docs/student-distill-design.md`, etapas E2/E3).

## Entorno

- Instancia Vast.ai, RTX 4090 de 48 GiB (47.38 GiB visibles), 256 GiB de disco.
- `scripts/bootstrap-vast.sh` con la release v2.1.0: dependencias fijadas,
  datasets restaurados y auditados, 33 archivos de pesos Würstchen descargados
  directo de HuggingFace con verificación SHA-256 completa, preflight CUDA ok.
- Checkpoint del maestro: `artifacts/lora-scale-3/pilot-checkpoint.pt`,
  SHA-256 `fbc61da942288997a59fecbc5fdf1ec2cdd88fcfbc4be88dee6617d45b013b47`,
  subido por `scp` y verificado en destino (< 200 MB, regla de transporte).

## Método

1. Captions deterministas en la instancia:
   `avatar-face generate-distill-captions` → 4096 pares (seed 42), splits
   train/validation/test 3276/410/410, SHA-256 del documento
   `04f75450b715eb500b0cdb994bf2f5e78b935b09412ef8c6bd4f5b19f0d54a0d` —
   **idéntico** al regenerado en la máquina local (determinismo verificado).
2. Smoke de 2 muestras con `scripts/generate_distill_dataset.py --limit 2`;
   inspección visual local: rostros de avatar adultos en vector plano con los
   atributos del caption (dentro del techo de fidelidad conocido del maestro).
3. Corrida completa reanudable (`records.jsonl`): receta oficial por muestra
   (1024 px, 30 timesteps de prior, guía 8.0, prompt negativo, 12 pasos de
   decoder, fp16, seed propia de 32 bits por par) y reducción a 256 px Lanczos.

## Resultados

- 4096/4096 muestras sin ninguna salida degenerada ni embedding no finito.
- Velocidad con pipeline caliente: ~2 s por muestra (~2.5 h de corrida),
  frente a las 9–15 h presupuestadas; costo real de E2 en el orden de 1–2 USD.
- Auditoría en la instancia: `approved: true`, 4096 hashes únicos, cero
  hallazgos (incluida la similitud perceptual).
- Release congelada: `data/distill-teacher-v1/dataset-v1.0.0.lock.json`,
  manifiesto SHA-256
  `05f36cb10f99efbbc4e34bcf36fa3274a8dcb0c9c348f3ed3fd6682df9300a26`.
- Paquete de bajada directa (sin Drive, regla 2026-08-19):
  `transfer/avatarface-distill-teacher-v1.tar` (154,613,760 bytes), SHA-256
  `ed0fc1d87e6e6f9f2c2919fa2266428ed3b5461e1d28b826d9d555196a5c0a1b`,
  verificado tras la descarga y con `verify-frozen-dataset` local.

## Decisión

La compuerta de E2 (dataset completo, auditado y congelado) queda superada.
E3 arranca en la misma instancia: `scripts/train_student.py --formulation
direct --batch-size 16` (batch 32 en fp32 excede la VRAM; con 16 usa
40.4 GiB), 50,000 pasos, lr 1e-4, EMA, muestras de control cada 5,000 pasos.
