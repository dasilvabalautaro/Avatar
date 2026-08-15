#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from run_wuerstchen_lora_pilot import load_text_model
from transformers import AutoTokenizer, CLIPTextModel

DEFAULT_PROMPT = (
    "flat vector avatar face, happy expression, square face, porcelain skin tone, "
    "side-parted black hair, green eyes with earrings, sky background"
)
DEFAULT_NEGATIVE_PROMPT = (
    "bad anatomy, blurry, fuzzy, extra arms, extra fingers, poorly drawn hands, "
    "disfigured, tiling, deformed, mutated, drawing"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--decoder-steps", type=int, default=12)
    parser.add_argument("--prior-guidance-scale", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    args = parser.parse_args()
    if not args.base_only and args.checkpoint is None:
        parser.error("--checkpoint es obligatorio salvo cuando se usa --base-only")
    if args.resolution < 256 or args.resolution % 8:
        parser.error("--resolution debe ser >= 256 y múltiplo de 8")

    root = args.root
    model_root = root / "models" / "wuerstchen-v2"
    device = torch.device("cuda")
    dtype = torch.float16
    text_encoder, tokenizer = load_text_model(model_root / "text-encoder", device, dtype)
    prior = WuerstchenPrior.from_pretrained(model_root / "prior-base", local_files_only=True).to(
        device=device, dtype=dtype
    )
    if not args.base_only:
        prior.add_adapter(
            LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            )
        )
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        set_peft_model_state_dict(prior, state["lora"])
    prior.eval()
    prior_pipe = WuerstchenPriorPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prior=prior,
        scheduler=DDPMWuerstchenScheduler(),
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode():
        prior_output = prior_pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.resolution,
            width=args.resolution,
            timesteps=DEFAULT_STAGE_C_TIMESTEPS,
            guidance_scale=args.prior_guidance_scale,
            generator=generator,
            output_type="pt",
        )
        embeddings = prior_output.image_embeddings
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("El prior produjo embeddings no finitos")
        print(
            "prior_ok "
            f"shape={tuple(embeddings.shape)} min={embeddings.min().item():.6f} "
            f"max={embeddings.max().item():.6f} mean={embeddings.mean().item():.6f} "
            f"std={embeddings.std().item():.6f}",
            flush=True,
        )

        prior_pipe.to("cpu")
        del prior_pipe, prior, text_encoder, tokenizer
        torch.cuda.empty_cache()

        decoder_root = model_root / "decoder"
        decoder_tokenizer = AutoTokenizer.from_pretrained(
            decoder_root / "tokenizer", local_files_only=True
        )
        decoder_text = CLIPTextModel.from_pretrained(
            decoder_root / "text_encoder", local_files_only=True
        ).to(device=device, dtype=dtype)
        decoder_model = WuerstchenDiffNeXt.from_pretrained(
            decoder_root / "decoder", local_files_only=True
        ).to(device=device, dtype=dtype)
        vqgan = PaellaVQModel.from_pretrained(decoder_root / "vqgan", local_files_only=True).to(
            device=device, dtype=dtype
        )
        decoder = WuerstchenDecoderPipeline(
            tokenizer=decoder_tokenizer,
            text_encoder=decoder_text,
            decoder=decoder_model,
            scheduler=DDPMWuerstchenScheduler.from_pretrained(
                decoder_root / "scheduler", local_files_only=True
            ),
            vqgan=vqgan,
        ).to(device)
        image = decoder(
            image_embeddings=embeddings,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.decoder_steps,
            guidance_scale=0.0,
            generator=generator,
            output_type="pil",
        ).images[0]

    pixels = np.asarray(image, dtype=np.float32)
    pixel_std = float(pixels.std())
    if not np.isfinite(pixels).all() or pixel_std < 2.0:
        raise RuntimeError(f"Salida visual degenerada: pixel_std={pixel_std:.6f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(
        f"validation_ok output={args.output} size={image.size} "
        f"pixel_mean={pixels.mean():.6f} pixel_std={pixel_std:.6f}"
    )


if __name__ == "__main__":
    main()
