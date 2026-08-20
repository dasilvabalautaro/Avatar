#!/usr/bin/env python3
"""Entrena el estudiante compacto del ADR 0010 sobre el dataset de destilación.

Formulaciones (`--formulation`):
- `direct` (primera): el estudiante aprende (ruido, atributos) → imagen del
  maestro con pérdida L1 e inferencia en 1 paso. Cada caption del dataset es
  único, así que el mapeo es determinista por atributos; el ruido de entrada
  queda como entrada de compatibilidad y el modelo aprende a ignorarlo.
- `diffusion` (respaldo): el mismo U-Net como predictor de epsilon con schedule
  coseno; muestreo DDIM de 8 pasos. **Descartada el 2026-08-19**: a ruido alto
  la predicción óptima de epsilon es la propia entrada ruidosa, así que el
  objetivo no da gradiente para usar el condicionamiento en los primeros pasos
  —los que fijan color y estructura global— y el muestreo desde ruido puro
  produce rostros correctos en forma pero lavados en color.
- `vpred` (vigente): mismo U-Net con parametrización **v** de Salimans & Ho,
  `v = sqrt(ab)·eps − sqrt(1−ab)·x0`. A ruido alto el objetivo equivale a la
  imagen limpia, de modo que el condicionamiento manda desde el primer paso y
  la proyección a x0 es estable en ambos extremos; muestreo DDIM de 8 pasos.

Reanudable con `--resume`: el checkpoint guarda pesos, EMA, optimizador, paso
y configuración completa. Cada `--sample-every` pasos genera las 8 muestras de
control (primeros 8 registros del split validation) con los pesos EMA.

Uso en la instancia (tras scripts/bootstrap-vast.sh y el dataset restaurado o
generado con scripts/generate_distill_dataset.py):
  python scripts/train_student.py --root /workspace/AvatarFace \
      --dataset-dir data/distill-teacher-v1 --formulation direct \
      --output artifacts/student-direct-1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image

from avatar_face.domain.attributes import LEGACY_ATTRIBUTES, AvatarAttributes
from avatar_face.infrastructure.training.student_unet import StudentUNet, StudentUNetConfig

CONTROL_SAMPLES = 8
EMA_DECAY = 0.999
DEFAULT_DDIM_STEPS = 8


def cosine_alpha_bar(t: torch.Tensor) -> torch.Tensor:
    """ᾱ(t) coseno (Nichol & Dhariwal), t en [0, 1]."""
    s = 0.008
    return torch.cos((t + s) / (1 + s) * math.pi / 2).clamp(1e-4, 0.9999) ** 2


def load_split(dataset_dir: Path, split: str) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Imágenes uint8 (n, 3, s, s) y atributos (n, 9) del split pedido."""
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [r for r in manifest["samples"] if r["split"] == split]
    if not records:
        raise SystemExit(f"El manifiesto no tiene muestras del split {split}")
    images = []
    attributes = []
    for record in records:
        with Image.open(dataset_dir / record["image"]) as image:
            array = torch.frombuffer(
                bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8
            ).reshape(image.height, image.width, 3)
        images.append(array.permute(2, 0, 1))
        attrs = AvatarAttributes(**{k: record["attributes"][k] for k in LEGACY_ATTRIBUTES})
        attributes.append(torch.tensor(attrs.indices(), dtype=torch.long))
    return torch.stack(images), torch.stack(attributes), records


def to_model_range(batch: torch.Tensor) -> torch.Tensor:
    return batch.float().div(127.5).sub(1.0)


def to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


@torch.no_grad()
def generate(
    model: StudentUNet,
    attributes: torch.Tensor,
    formulation: str,
    image_size: int,
    generator: torch.Generator,
    device: torch.device,
    ddim_steps: int = DEFAULT_DDIM_STEPS,
) -> torch.Tensor:
    """Inferencia del estudiante: 1 paso (direct) o DDIM de 8 pasos (diffusion)."""
    batch = attributes.shape[0]
    x = torch.randn(
        (batch, 3, image_size, image_size), generator=generator, device=device
    )
    if formulation == "direct":
        ratio = torch.zeros(batch, device=device)
        return model(x, ratio, attributes)
    times = torch.linspace(1.0, 0.0, ddim_steps + 1, device=device)
    for index in range(ddim_steps):
        t, t_next = times[index], times[index + 1]
        ab_t = cosine_alpha_bar(t)
        ab_next = cosine_alpha_bar(t_next)
        prediction = model(x, t.expand(batch), attributes)
        if formulation == "vpred":
            # x0 y eps se despejan de v sin dividir por sqrt(ab): estable en t→1.
            x0 = ab_t.sqrt() * x - (1 - ab_t).sqrt() * prediction
            epsilon = (1 - ab_t).sqrt() * x + ab_t.sqrt() * prediction
        else:
            epsilon = prediction
            x0 = (x - (1 - ab_t).sqrt() * epsilon) / ab_t.sqrt()
        x = ab_next.sqrt() * x0.clamp(-1, 1) + (1 - ab_next).sqrt() * epsilon
    return x


def save_control_samples(
    model: StudentUNet,
    attributes: torch.Tensor,
    records: list[dict],
    formulation: str,
    image_size: int,
    directory: Path,
    step: int,
    device: torch.device,
    ddim_steps: int,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(42)
    model.eval()
    outputs = generate(
        model, attributes, formulation, image_size, generator, device, ddim_steps
    )
    model.train()
    for record, output in zip(records, outputs, strict=True):
        to_pil(output).save(directory / f"step-{step:06d}-{record['identifier']}.png")
    print(f"control_samples step={step} dir={directory}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/distill-teacher-v1"))
    parser.add_argument(
        "--formulation", choices=("direct", "diffusion", "vpred"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--sample-every", type=int, default=5_000)
    parser.add_argument("--base-channels", type=int, default=96)
    parser.add_argument(
        "--attention-resolutions",
        default="32",
        help="resoluciones con atención, separadas por comas (32 en el modelo base; 16 en "
        "las variantes ligeras, donde la atención a 32 px domina el coste móvil)",
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=DEFAULT_DDIM_STEPS,
        help="pasos de muestreo de las muestras de control y del contrato móvil",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dataset_dir = (args.root / args.dataset_dir).resolve()
    output_dir = (args.root / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "student-checkpoint.pt"

    train_images, train_attributes, _ = load_split(dataset_dir, "train")
    eval_images, eval_attributes, eval_records = load_split(dataset_dir, "validation")
    eval_attributes = eval_attributes[:CONTROL_SAMPLES].to(device)
    eval_records = eval_records[:CONTROL_SAMPLES]
    print(
        f"train={train_images.shape[0]} eval={len(eval_records)} "
        f"image_size={train_images.shape[-1]}",
        flush=True,
    )
    if train_images.shape[-1] != args.image_size:
        raise SystemExit(
            f"El dataset es de {train_images.shape[-1]} px y se pidió {args.image_size} px"
        )

    config = StudentUNetConfig(
        image_size=args.image_size,
        base_channels=args.base_channels,
        attention_resolutions=tuple(
            int(x) for x in args.attention_resolutions.split(",") if x.strip()
        ),
    )
    student = StudentUNet(config).to(device)
    ema = StudentUNet(config).to(device)
    ema.load_state_dict(student.state_dict())
    ema.requires_grad_(False)
    print(f"parameters={student.parameter_count():,}", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    start_step = 0
    losses: list[float] = []
    if args.resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if state["train_config"]["formulation"] != args.formulation:
            raise SystemExit("El checkpoint es de otra formulación; no se puede reanudar.")
        student.load_state_dict(state["student"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["completed_steps"])
        losses = list(state["losses"])
        print(f"reanudando desde el paso {start_step}", flush=True)

    def learning_rate_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.learning_rate * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    def save_checkpoint(completed_steps: int) -> None:
        torch.save(
            {
                "schema_version": 1,
                "completed_steps": completed_steps,
                "seed": args.seed,
                "train_config": {
                    "formulation": args.formulation,
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "warmup_steps": args.warmup_steps,
                    "ema_decay": EMA_DECAY,
                    "ddim_steps": args.ddim_steps,
                    "dataset_dir": str(args.dataset_dir),
                },
                "student_config": {
                    "image_size": config.image_size,
                    "base_channels": config.base_channels,
                    "channel_multipliers": list(config.channel_multipliers),
                    "residual_blocks_per_level": config.residual_blocks_per_level,
                    "attention_resolutions": list(config.attention_resolutions),
                    "attention_heads": config.attention_heads,
                    "condition_dim": config.condition_dim,
                    "attribute_embedding_dim": config.attribute_embedding_dim,
                    "attribute_cardinalities": list(config.attribute_cardinalities),
                },
                "losses": losses,
                "student": student.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_path,
        )

    student.train()
    for step in range(start_step, args.steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step)
        indices = torch.randint(train_images.shape[0], (args.batch_size,))
        images = to_model_range(train_images[indices]).to(device)
        attributes = train_attributes[indices].to(device)
        if args.formulation == "direct":
            noise = torch.randn_like(images)
            ratio = torch.zeros(args.batch_size, device=device)
            prediction = student(noise, ratio, attributes)
            loss = torch.nn.functional.l1_loss(prediction, images)
        else:
            t = torch.rand(args.batch_size, device=device)
            ab = cosine_alpha_bar(t)[:, None, None, None]
            epsilon = torch.randn_like(images)
            noisy = ab.sqrt() * images + (1 - ab).sqrt() * epsilon
            prediction = student(noisy, t, attributes)
            if args.formulation == "vpred":
                target = ab.sqrt() * epsilon - (1 - ab).sqrt() * images
            else:
                target = epsilon
            loss = torch.nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(loss.detach()):
            raise RuntimeError(f"Pérdida no finita en el paso {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for ema_parameter, parameter in zip(
                ema.parameters(), student.parameters(), strict=True
            ):
                ema_parameter.lerp_(parameter, 1 - EMA_DECAY)
            for ema_buffer, buffer in zip(ema.buffers(), student.buffers(), strict=True):
                ema_buffer.copy_(buffer)
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 50 == 0 or step + 1 == args.steps:
            print(
                f"step={step + 1}/{args.steps} loss={losses[-1]:.6f} "
                f"lr={learning_rate_at(step):.2e}",
                flush=True,
            )
        if (step + 1) % args.sample_every == 0 or step + 1 == args.steps:
            save_checkpoint(step + 1)
            save_control_samples(
                ema,
                eval_attributes,
                eval_records,
                args.formulation,
                args.image_size,
                output_dir / "control",
                step + 1,
                device,
                args.ddim_steps,
            )

    save_checkpoint(args.steps)
    print(
        f"train_ok formulation={args.formulation} steps={args.steps} "
        f"final_loss={losses[-1]:.6f} checkpoint={checkpoint_path}"
    )


if __name__ == "__main__":
    main()
