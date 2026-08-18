#!/usr/bin/env python3
"""Destilación progresiva del prior Würstchen Stage C (Salimans & Ho).

Implementa docs/distill-prior-design.md: el estudiante aprende en UN paso lo
que el maestro (con guía classifier-free ya combinada) hace en DOS. La
destilación ocurre en el espacio del prior (texto → embeddings de imagen
16×8×8 a 256 px), así que sólo se necesitan captions, no imágenes.

Rejillas (verificadas contra diffusers==0.39.0):
- Etapa 1: maestro = rejilla oficial DEFAULT_STAGE_C_TIMESTEPS (30 puntos),
  estudiante = puntos 0, 2, ..., 28 (15 puntos).
- Etapa 2: maestro = rejilla del estudiante de la etapa 1 (15 puntos),
  estudiante = puntos 0, 4, ..., 28 de la oficial (8 puntos).
En ambas, el salto del estudiante k va del punto 2k al 2k+2 del maestro.

Matemática por paso (schedulers/scheduling_ddpm_wuerstchen.py):
- ab(t) = cos²((t+s)/(1+s)·π/2) / ab(0), s=0.008, fijada a [1e-4, 0.9999].
- Un paso del scheduler desde t hasta t' es lineal en la predicción eps:
  x' = sqrt(1/alpha)·x_t − sqrt(1/alpha)·(1−alpha)/sqrt(1−ab(t))·eps + ruido,
  con alpha = ab(t)/ab(t').
- El maestro ejecuta dos pasos guiados desde t (punto 2k) hasta t'' (2k+2).
  El epsilon efectivo del salto t → t'' se despeja de los coeficientes del
  salto único: eps_obj = (a·x_t − x'') / b, con a = sqrt(ab(t'')/ab(t)) y
  b = a·(1 − ab(t)/ab(t'')) / sqrt(1 − ab(t)). El ruido estocástico de los
  pasos del maestro queda absorbido en ese epsilon efectivo.
- En modo `--mode marginal` (etapas 1 a 1c, hoy descartado), x_t se muestrea
  como N(0, I): equivale a add_noise con x0 ~ N(0, I), pues
  sqrt(ab)·z + sqrt(1−ab)·eps es N(0, I) para cualquier t. Es la marginal
  correcta del scheduler cuando no hay imágenes, pero el estudiante nunca ve
  los estados intermedios reales que recibirá en inferencia.
- En modo `--mode trajectory` (ADR 0009, docs/distill-trajectory-design.md) el
  maestro primero ejecuta su trayectoria guiada completa para cada caption de
  train y se almacenan los latentes de todos los puntos de la rejilla junto
  con los embeddings de texto; el entrenamiento usa los latentes reales
  x(2k) → x(2k+2) como entrada y objetivo, sin más forwards del maestro.
- Guía (pipeline_wuerstchen_prior.py): eps = lerp(eps_incond, eps_cond, w)
  = eps_incond + w·(eps_cond − eps_incond), con incond = caption vacío.
- Pérdida: MSE en fp32 con **peso uniforme** por salto. La ponderación SNR
  inicial (w = ab/(1-ab)) dejaba sin gradiente los saltos de alto ruido y la
  etapa 1 falló la compuerta; ver
  docs/experiments/wuerstchen-distill-stage-1-2026-08-18.md.
  Con `--normalize-target`, la pérdida de cada muestra se divide por la
  potencia media del epsilon objetivo (mse/‖eps‖²): todos los saltos pesan
  igual aunque la magnitud de su epsilon difiera en órdenes de magnitud
  (diagnóstico de la etapa 1b,
  docs/experiments/wuerstchen-distill-stage-1b-2026-08-18.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import DDPMWuerstchenScheduler
from diffusers.pipelines.deprecated.wuerstchen.modeling_wuerstchen_prior import WuerstchenPrior
from diffusers.pipelines.deprecated.wuerstchen.pipeline_wuerstchen_prior import (
    DEFAULT_STAGE_C_TIMESTEPS,
)
from peft import LoraConfig, set_peft_model_state_dict
from run_wuerstchen_lora_pilot import load_text_model

# Valores históricos de las escalas 1-3, por si el checkpoint no los guarda.
DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
}
LATENT_SHAPE = (1, 16, 8, 8)  # embeddings EffNet a 256 px (resolución de producto)


def teacher_grid(stage: int) -> list[float]:
    """Rejilla del maestro: la oficial de 30 en etapa 1; su mitad par en etapa 2."""
    if stage == 1:
        return [float(t) for t in DEFAULT_STAGE_C_TIMESTEPS]
    return [float(DEFAULT_STAGE_C_TIMESTEPS[i]) for i in range(0, 30, 2)]


def jump_coefficients(
    scheduler: DDPMWuerstchenScheduler, grid: list[float], device: torch.device
) -> tuple[list[tuple[float, float, float, float]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coeficientes fp32 de cada salto del estudiante k: t → t'' = puntos 2k → 2k+2.

    Devuelve la lista (t, a, b, peso_snr) y los tensores a, b y pesos apilados.
    """
    jumps: list[tuple[float, float, float, float]] = []
    for k in range(len(grid) // 2 - 1 + len(grid) % 2):
        t = torch.tensor([grid[2 * k]], dtype=torch.float32, device=device)
        t_next = torch.tensor([grid[2 * k + 2]], dtype=torch.float32, device=device)
        ab_t = scheduler._alpha_cumprod(t, device)
        ab_next = scheduler._alpha_cumprod(t_next, device)
        alpha = ab_t / ab_next
        a = alpha.rsqrt()  # sqrt(ab(t'')/ab(t))
        b = a * (1 - alpha) / (1 - ab_t).sqrt()
        jumps.append((float(t), float(a), float(b), 1.0))
    coeffs_a = torch.tensor([j[1] for j in jumps], dtype=torch.float32, device=device)
    coeffs_b = torch.tensor([j[2] for j in jumps], dtype=torch.float32, device=device)
    weights = torch.tensor([j[3] for j in jumps], dtype=torch.float32, device=device)
    return jumps, coeffs_a, coeffs_b, weights


def encode_captions(
    text_model: torch.nn.Module, tokenizer: object, captions: list[str], device: torch.device
) -> torch.Tensor:
    tokens = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        return text_model(tokens.input_ids, attention_mask=tokens.attention_mask).last_hidden_state


def guided_epsilon(
    model: torch.nn.Module,
    x: torch.Tensor,
    ratio: torch.Tensor,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Guía classifier-free exactamente como la pipeline: lerp(incond, cond, w)."""
    both = model(
        torch.cat([x, x]), r=torch.cat([ratio, ratio]), c=torch.cat([cond, uncond])
    )
    eps_cond, eps_uncond = both.chunk(2)
    return torch.lerp(eps_uncond, eps_cond, guidance_scale)


def generate_trajectories(
    teacher: torch.nn.Module,
    text_model: torch.nn.Module,
    tokenizer: object,
    captions: list[str],
    grid: list[float],
    scheduler: DDPMWuerstchenScheduler,
    uncond: torch.Tensor,
    guidance_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trayectorias reales del maestro: latentes en cada punto de la rejilla.

    Devuelve (trayectorias, condiciones): un tensor
    [n_captions, len(grid), 16, 8, 8] con el latente real del maestro en cada
    punto de su rejilla, y un tensor [n_captions, seq, hidden] con los
    embeddings de texto cacheados. El entrenamiento consume ambos sin volver a
    ejecutar el maestro ni el encoder de texto.
    """
    conditions: list[torch.Tensor] = []
    trajectories = torch.empty(
        len(captions), len(grid), *LATENT_SHAPE[1:], dtype=dtype, device=device
    )
    with torch.no_grad():
        for i, caption in enumerate(captions):
            cond = encode_captions(text_model, tokenizer, [caption], device)
            conditions.append(cond[0])
            x = torch.randn(LATENT_SHAPE, device=device, dtype=dtype)
            for j, t in enumerate(grid):
                trajectories[i, j] = x[0]
                if j + 1 == len(grid):
                    break
                ratio = torch.full((1,), t, device=device, dtype=dtype)
                eps = guided_epsilon(teacher, x, ratio, cond, uncond, guidance_scale)
                x = scheduler.step(eps, ratio, x).prev_sample
            if (i + 1) % 50 == 0 or i + 1 == len(captions):
                print(f"trajectories={i + 1}/{len(captions)}", flush=True)
    return trajectories, torch.stack(conditions)


def load_prior(model_root: Path, device: torch.device, dtype: torch.dtype) -> WuerstchenPrior:
    return WuerstchenPrior.from_pretrained(model_root / "prior-base", local_files_only=True).to(
        device=device, dtype=dtype
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--student-steps", type=int, required=True, help="15 en etapa 1, 8 en etapa 2"
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--mode",
        choices=("marginal", "trajectory"),
        default="marginal",
        help="marginal: x_t ~ N(0, I) (etapas 1-1c, descartado); trajectory: latentes "
        "reales de la trayectoria del maestro (ADR 0009)",
    )
    parser.add_argument(
        "--normalize-target",
        action="store_true",
        help="divide la pérdida de cada muestra por la potencia media del epsilon objetivo",
    )
    parser.add_argument("--guidance-scale", type=float, default=8.0)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/training-procedural-v2-1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda")
    teacher_dtype = torch.bfloat16
    root = args.root
    dataset_dir = root / args.dataset_dir
    model_root = root / "models" / "wuerstchen-v2"

    grid = teacher_grid(args.stage)
    student_steps = len(grid[::2])
    if args.student_steps != student_steps:
        parser.error(
            f"--student-steps debe ser {student_steps} para la etapa {args.stage} "
            f"(rejilla del maestro de {len(grid)} puntos)"
        )
    jumps, coeffs_a, coeffs_b, weights = jump_coefficients(
        DDPMWuerstchenScheduler(), grid, device
    )
    print(f"grid_teacher={len(grid)} grid_student={student_steps} jumps={len(jumps)}", flush=True)

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    train = [x for x in manifest["samples"] if x["split"] == "train"]
    if not train:
        parser.error(f"El manifiesto {dataset_dir / 'manifest.json'} no tiene split train")

    text_model, tokenizer = load_text_model(model_root / "text-encoder", device, teacher_dtype)
    uncond = encode_captions(text_model, tokenizer, [""], device)

    state = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=True)
    teacher = load_prior(model_root, device, teacher_dtype)
    student = load_prior(model_root, device, torch.float32)
    if args.stage == 1:
        if "lora" not in state:
            parser.error("La etapa 1 requiere un checkpoint LoRA (clave 'lora')")
        lora_config = state.get("lora_config", DEFAULT_LORA_CONFIG)
        teacher.add_adapter(LoraConfig(**lora_config))
        set_peft_model_state_dict(teacher, state["lora"])
        # Estudiante: copia del prior con el LoRA fusionado, sin restos peft.
        student.add_adapter(LoraConfig(**lora_config))
        set_peft_model_state_dict(student, {k: v.float() for k, v in state["lora"].items()})
        student.fuse_lora()
        student.unload_lora()
    else:
        if "prior" not in state:
            parser.error("La etapa 2 requiere el checkpoint completo de la etapa 1 (clave 'prior')")
        teacher.load_state_dict(state["prior"], strict=True)
        student.load_state_dict({k: v.float() for k, v in state["prior"].items()}, strict=True)
    teacher.eval().requires_grad_(False)
    student.train().requires_grad_(True)

    scheduler = DDPMWuerstchenScheduler()
    scheduler.set_timesteps(timesteps=grid, device=device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)

    trajectories: torch.Tensor | None = None
    conditions: torch.Tensor | None = None
    if args.mode == "trajectory":
        trajectories, conditions = generate_trajectories(
            teacher,
            text_model,
            tokenizer,
            [record["caption"] for record in train],
            grid,
            scheduler,
            uncond,
            args.guidance_scale,
            device,
            teacher_dtype,
        )
        # El maestro ya no participa en el entrenamiento: se libera VRAM.
        del teacher
        torch.cuda.empty_cache()

    losses: list[float] = []
    for step in range(args.steps):
        index = step % len(train)
        k = int(torch.randint(len(jumps), (1,)).item())
        t = grid[2 * k]
        if trajectories is None or conditions is None:
            record = train[index]
            cond = encode_captions(text_model, tokenizer, [record["caption"]], device)
            t_mid = grid[2 * k + 1]
            x_t = torch.randn(LATENT_SHAPE, device=device, dtype=teacher_dtype)
            ratio = torch.full((1,), t, device=device, dtype=teacher_dtype)
            ratio_mid = torch.full((1,), t_mid, device=device, dtype=teacher_dtype)
            with torch.no_grad():
                eps1 = guided_epsilon(teacher, x_t, ratio, cond, uncond, args.guidance_scale)
                x_mid = scheduler.step(eps1, ratio, x_t).prev_sample
                eps2 = guided_epsilon(teacher, x_mid, ratio_mid, cond, uncond, args.guidance_scale)
                x_next = scheduler.step(eps2, ratio_mid, x_mid).prev_sample
        else:
            # Entrada y objetivo reales de la trayectoria del maestro (ADR 0009).
            cond = conditions[index].unsqueeze(0)
            x_t = trajectories[index, 2 * k].unsqueeze(0)
            x_next = trajectories[index, 2 * k + 2].unsqueeze(0)
        # Epsilon efectivo del salto t → t'' despejado de los coeficientes.
        target = (coeffs_a[k] * x_t.float() - x_next.float()) / coeffs_b[k]
        ratio_student = torch.full((1,), t, device=device, dtype=torch.float32)
        prediction = student(x_t.float(), ratio_student, cond.float())
        loss = torch.nn.functional.mse_loss(prediction.float(), target) * weights[k]
        if args.normalize_target:
            loss = loss / target.pow(2).mean().clamp_min(1e-8)
        if not torch.isfinite(loss.detach()).all():
            raise RuntimeError(
                f"Pérdida no finita en el paso {step + 1} (k={k}, t={t}): "
                f"{float(loss.detach().cpu())}"
            )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        print(f"step={step + 1}/{args.steps} loss={loss_value:.6f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "pilot-checkpoint.pt"
    torch.save(
        {
            "steps": args.steps,
            "seed": 42,
            "distill_config": {
                "stage": args.stage,
                "student_steps": args.student_steps,
                "guidance_scale": args.guidance_scale,
                "teacher_steps": len(grid),
                "normalize_target": args.normalize_target,
                "learning_rate": args.learning_rate,
                "mode": args.mode,
            },
            "losses": losses,
            "prior": {
                k: v.detach().to(torch.bfloat16).cpu() for k, v in student.state_dict().items()
            },
        },
        checkpoint,
    )
    print(f"distill_ok steps={args.steps} final_loss={losses[-1]:.6f} checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
