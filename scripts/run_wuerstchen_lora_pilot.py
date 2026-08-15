#!/usr/bin/env python3
"""Piloto LoRA autocontenido para el prior Würstchen Stage C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import DDPMWuerstchenScheduler
from diffusers.pipelines.deprecated.wuerstchen.modeling_wuerstchen_prior import WuerstchenPrior
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModel


def load_text_model(
    directory: Path, device: torch.device, dtype: torch.dtype
) -> tuple[CLIPTextModel, object]:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    text_config = CLIPTextConfig(**config["text_config"])
    model = CLIPTextModel(text_config)
    state: dict[str, torch.Tensor] = {}
    for path in sorted(directory.glob("pytorch_model-*.safetensors")):
        state.update(
            {
                key: value
                for key, value in load_file(str(path)).items()
                if key.startswith("text_model.")
            }
        )
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    return model, tokenizer


def load_effnet(stage_b: Path, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = efficientnet_v2_s(weights=None).features
            self.mapper = torch.nn.Sequential(
                torch.nn.Conv2d(1280, 16, kernel_size=1, bias=False), torch.nn.BatchNorm2d(16)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.mapper(self.backbone(x))

    encoder = Encoder()
    state = torch.load(stage_b, map_location="cpu", weights_only=False)
    encoder.load_state_dict(state["effnet_state_dict"])
    del state
    encoder.eval().requires_grad_(False).to(device=device, dtype=dtype)
    return encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    root = args.root
    model_root = root / "models" / "wuerstchen-v2"
    manifest = json.loads((root / "data/training-procedural-v2/manifest.json").read_text())
    train = [x for x in manifest["samples"] if x["split"] == "train"][: max(args.steps, 20)]
    tokenizer_model, tokenizer = load_text_model(model_root / "text-encoder", device, dtype)
    image_encoder = load_effnet(model_root / "stage-b-encoder/model_v2_stage_b.pt", device, dtype)
    prior = WuerstchenPrior.from_pretrained(model_root / "prior-base", local_files_only=True).to(
        device=device, dtype=dtype
    )
    prior.requires_grad_(False)
    prior.add_adapter(
        LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    optimizer = torch.optim.AdamW(
        [p for p in prior.parameters() if p.requires_grad], lr=args.learning_rate
    )
    scheduler = DDPMWuerstchenScheduler()
    image_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    losses: list[float] = []
    prior.train()
    for step in range(args.steps):
        record = train[step % len(train)]
        image = (
            image_transform(
                Image.open(root / "data/training-procedural-v2" / record["image"]).convert("RGB")
            )
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
        )
        tokens = tokenizer(
            [record["caption"]],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            cond = tokenizer_model(
                tokens.input_ids, attention_mask=tokens.attention_mask
            ).last_hidden_state
            target = image_encoder(image).add(1).div(42)
            noise = torch.randn_like(target)
            timestep = torch.rand((1,), device=device, dtype=dtype)
            noisy = scheduler.add_noise(target, noise, timestep)
        prediction = prior(noisy, timestep, cond)
        loss = torch.nn.functional.mse_loss(prediction.float(), noise.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        print(f"step={step + 1}/{args.steps} loss={losses[-1]:.6f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "steps": args.steps,
            "seed": 42,
            "losses": losses,
            "lora": get_peft_model_state_dict(prior),
        },
        args.output / "pilot-checkpoint.pt",
    )
    print(
        f"pilot_ok steps={args.steps} final_loss={losses[-1]:.6f} "
        f"checkpoint={args.output / 'pilot-checkpoint.pt'}"
    )


if __name__ == "__main__":
    main()
