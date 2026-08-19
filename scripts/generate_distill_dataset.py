#!/usr/bin/env python3
"""Genera el dataset de destilación del ADR 0010: pares caption → imagen del maestro.

El maestro (prior base + LoRA escala-3, receta oficial: 1024 px, 30 timesteps,
guía 8.0, prompt negativo, fp16) genera una imagen por cada par (caption, seed)
del documento producido por `avatar-face generate-distill-captions`; la imagen
se reduce a 256 px (Lanczos, resolución de producto) y se registra con SHA-256.

Reanudable: cada muestra completada se anota en `records.jsonl`; al relanzar,
las muestras ya registradas cuyo archivo conserva su hash se saltan. Al
terminar todos los pares se escribe `manifest.json` (schema 1, fuente
`AF-DISTILL-001`, CC0-1.0) compatible con `avatar-face audit-dataset` y
`freeze-dataset`.

Uso en la instancia (tras scripts/bootstrap-vast.sh):
  avatar-face generate-distill-captions --output artifacts/distill/captions-v1.json
  python scripts/generate_distill_dataset.py --root /workspace/AvatarFace \
      --teacher-checkpoint <checkpoint LoRA escala-3> \
      --captions artifacts/distill/captions-v1.json \
      --output-dir data/distill-teacher-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMWuerstchenScheduler
from diffusers.pipelines.deprecated.wuerstchen.modeling_paella_vq_model import PaellaVQModel
from diffusers.pipelines.deprecated.wuerstchen.modeling_wuerstchen_diffnext import (
    WuerstchenDiffNeXt,
)
from diffusers.pipelines.deprecated.wuerstchen.modeling_wuerstchen_prior import WuerstchenPrior
from diffusers.pipelines.deprecated.wuerstchen.pipeline_wuerstchen import WuerstchenDecoderPipeline
from diffusers.pipelines.deprecated.wuerstchen.pipeline_wuerstchen_prior import (
    DEFAULT_STAGE_C_TIMESTEPS,
    WuerstchenPriorPipeline,
)
from peft import LoraConfig, set_peft_model_state_dict
from PIL import Image
from run_wuerstchen_lora_pilot import load_text_model
from transformers import AutoTokenizer, CLIPTextModel

DEFAULT_NEGATIVE_PROMPT = (
    "child, kid, teenager, baby, minor, underage, bad anatomy, blurry, fuzzy, extra arms, "
    "extra fingers, poorly drawn hands, disfigured, tiling, deformed, mutated, drawing"
)
DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/distill-teacher-v1"))
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--final-size", type=int, default=256)
    parser.add_argument("--decoder-steps", type=int, default=12)
    parser.add_argument("--prior-guidance-scale", type=float, default=8.0)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--limit", type=int, default=0, help="0 = todos los pares pendientes")
    args = parser.parse_args()

    root = args.root
    model_root = root / "models" / "wuerstchen-v2"
    output_dir = (root / args.output_dir).resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    manifest_path = output_dir / "manifest.json"

    captions_path = args.captions if args.captions.is_absolute() else root / args.captions
    captions_bytes = captions_path.read_bytes()
    captions_sha256 = hashlib.sha256(captions_bytes).hexdigest()
    payload = json.loads(captions_bytes.decode("utf-8"))
    pairs = payload["pairs"]
    print(f"captions={len(pairs)} sha256={captions_sha256}", flush=True)

    completed: dict[str, dict] = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            image_path = output_dir / record["image"]
            if image_path.is_file() and sha256_of(image_path) == record["sha256"]:
                completed[record["identifier"]] = record
        print(f"reanudando: {len(completed)} muestras ya completadas", flush=True)

    pending = [pair for pair in pairs if pair["identifier"] not in completed]
    if args.limit:
        pending = pending[: args.limit]

    if pending:
        device = torch.device("cuda")
        dtype = torch.float16
        text_encoder, tokenizer = load_text_model(model_root / "text-encoder", device, dtype)
        prior = WuerstchenPrior.from_pretrained(
            model_root / "prior-base", local_files_only=True
        ).to(device=device, dtype=dtype)
        state = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=True)
        lora_config = state.get("lora_config", DEFAULT_LORA_CONFIG)
        prior.add_adapter(LoraConfig(**lora_config))
        set_peft_model_state_dict(prior, state["lora"])
        prior.eval()
        prior_pipe = WuerstchenPriorPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prior=prior,
            scheduler=DDPMWuerstchenScheduler(),
        ).to(device)

        decoder_root = model_root / "decoder"
        decoder = WuerstchenDecoderPipeline(
            tokenizer=AutoTokenizer.from_pretrained(
                decoder_root / "tokenizer", local_files_only=True
            ),
            text_encoder=CLIPTextModel.from_pretrained(
                decoder_root / "text_encoder", local_files_only=True
            ).to(device=device, dtype=dtype),
            decoder=WuerstchenDiffNeXt.from_pretrained(
                decoder_root / "decoder", local_files_only=True
            ).to(device=device, dtype=dtype),
            scheduler=DDPMWuerstchenScheduler.from_pretrained(
                decoder_root / "scheduler", local_files_only=True
            ),
            vqgan=PaellaVQModel.from_pretrained(decoder_root / "vqgan", local_files_only=True).to(
                device=device, dtype=dtype
            ),
        ).to(device)

        with records_path.open("a", encoding="utf-8") as records_stream:
            for position, pair in enumerate(pending):
                generator = torch.Generator(device=device).manual_seed(int(pair["seed"]))
                with torch.inference_mode():
                    embeddings = prior_pipe(
                        prompt=pair["caption"],
                        negative_prompt=args.negative_prompt,
                        height=args.resolution,
                        width=args.resolution,
                        timesteps=DEFAULT_STAGE_C_TIMESTEPS,
                        guidance_scale=args.prior_guidance_scale,
                        generator=generator,
                        output_type="pt",
                    ).image_embeddings
                    if not torch.isfinite(embeddings).all():
                        raise RuntimeError(f"Embeddings no finitos en {pair['identifier']}")
                    image = decoder(
                        image_embeddings=embeddings,
                        prompt=pair["caption"],
                        negative_prompt=args.negative_prompt,
                        num_inference_steps=args.decoder_steps,
                        guidance_scale=0.0,
                        generator=generator,
                        output_type="pil",
                    ).images[0]
                pixels = np.asarray(image, dtype=np.float32)
                if not np.isfinite(pixels).all() or float(pixels.std()) < 2.0:
                    raise RuntimeError(
                        f"Salida degenerada en {pair['identifier']}: std={pixels.std():.4f}"
                    )
                reduced = image.resize(
                    (args.final_size, args.final_size), Image.Resampling.LANCZOS
                )
                image_path = images_dir / f"{pair['identifier']}.png"
                reduced.save(image_path, format="PNG", optimize=True)
                record = {
                    "identifier": pair["identifier"],
                    "image": f"images/{image_path.name}",
                    "caption": pair["caption"],
                    "attributes": pair["attributes"],
                    "source": "AF-DISTILL-001",
                    "creator": "AvatarFace project",
                    "license_id": "CC0-1.0",
                    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "consent_or_release": "not_applicable_synthetic",
                    "sha256": sha256_of(image_path),
                    "split": pair["split"],
                    "synthetic": True,
                    "sample_seed": int(pair["seed"]),
                }
                records_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_stream.flush()
                completed[record["identifier"]] = record
                print(
                    f"sample={position + 1}/{len(pending)} id={pair['identifier']} "
                    f"total={len(completed)}/{len(pairs)}",
                    flush=True,
                )

    if len(completed) < len(pairs):
        print(
            f"parcial: {len(completed)}/{len(pairs)} muestras; relanzar para continuar",
            flush=True,
        )
        return

    manifest = {
        "schema_version": 1,
        "dataset": {
            "name": "avatarface-distill-teacher",
            "version": "1.0.0",
            "generator": "wuerstchen-lora-official-recipe-v1",
            "seed": payload["seed"],
            "image_size": args.final_size,
            "license_id": "CC0-1.0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "contains_real_people": False,
            "uses_external_assets": False,
            "teacher": {
                "checkpoint_sha256": sha256_of(args.teacher_checkpoint),
                "captions_sha256": captions_sha256,
                "prior_timesteps": len(DEFAULT_STAGE_C_TIMESTEPS),
                "decoder_steps": args.decoder_steps,
                "prior_guidance_scale": args.prior_guidance_scale,
                "resolution": args.resolution,
                "negative_prompt": args.negative_prompt,
            },
        },
        "samples": [completed[pair["identifier"]] for pair in pairs],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"dataset_ok samples={len(pairs)} manifest={manifest_path} "
        f"manifest_sha256={sha256_of(manifest_path)}"
    )


if __name__ == "__main__":
    main()
