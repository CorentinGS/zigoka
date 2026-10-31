"""go-train: distill teacher policy+value into the student.

Loss = (1-hard_mix) * soft CE + hard_mix * hard CE + value_weight * value MSE,
the mixed hard/soft recipe Moka's sweep selected. Whole-game validation
split, dihedral augmentation, AMP + pinned loaders on NVIDIA CUDA.
Checkpoint selection scores the INT8-quantized snapshot (what the shipped
artifact actually computes), not the float weights. Training is
deterministic by default (seeded per-item augmentation, no cuDNN
benchmark); pass --non-deterministic for the faster nondeterministic path.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import GoDataset, load_npz, split_by_game
from .device import amp_dtype, configure, describe, is_cuda, resolve_device
from .manifest import import_model, quantized_snapshot
from .model import MokaNet, load_checkpoint, save_checkpoint


@dataclass
class TrainConfig:
    data: str
    out: str = "checkpoints/student.pt"
    init_checkpoint: str | None = None
    init_dir: str | None = None
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    hard_mix: float = 0.5
    value_weight: float = 1.0
    val_fraction: float = 0.1
    seed: int = 1
    num_workers: int = 2
    device: str = "auto"
    amp: bool = True
    deterministic: bool = True


def _run_epoch(model, loader, device, dtype, optimizer=None, scaler=None, hard_mix=0.5, value_weight=1.0):
    training = optimizer is not None
    model.train(training)
    loss_sum, agree, value_mae, count = 0.0, 0, 0.0, 0
    for feats, policy, value in loader:
        feats = feats.to(device, non_blocking=True)
        policy = policy.to(device, non_blocking=True)
        value = value.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            logits, pred_value = model(feats)
            log_probs = torch.log_softmax(logits, dim=1)
            soft_ce = -(policy * log_probs).sum(1).mean()
            hard_ce = torch.nn.functional.nll_loss(log_probs, policy.argmax(1))
            loss = (1 - hard_mix) * soft_ce + hard_mix * hard_ce + value_weight * torch.nn.functional.mse_loss(pred_value, value)
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        batch = len(value)
        loss_sum += float(loss) * batch
        agree += int((logits.argmax(1) == policy.argmax(1)).sum())
        value_mae += float((pred_value - value).abs().sum())
        count += batch
    if count == 0:
        raise ValueError("epoch ran over an empty loader")
    return {"loss": loss_sum / count, "top_move_agreement": agree / count, "value_mae": value_mae / count}


def train(config: TrainConfig) -> Path:
    device = resolve_device(config.device)
    configure(device, deterministic=config.deterministic)
    dtype = amp_dtype(device) if config.amp else None
    print(f"device: {describe(device)}; amp: {dtype}; deterministic: {config.deterministic}")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if is_cuda(device):
        torch.cuda.manual_seed_all(config.seed)

    data = load_npz(config.data)
    train_idx, val_idx = split_by_game(data, config.val_fraction, config.seed)
    pin = is_cuda(device)
    train_set = GoDataset(data, train_idx, augment=True, seed=config.seed)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=pin)
    val_loader = DataLoader(GoDataset(data, val_idx), batch_size=config.batch_size,
                            num_workers=config.num_workers, pin_memory=pin) if len(val_idx) else None
    if config.init_checkpoint:
        model = load_checkpoint(config.init_checkpoint, device)
    elif config.init_dir:
        model = import_model(config.init_dir, device)
    else:
        model = MokaNet().to(device)
    print(f"student parameters: {model.parameter_count():,}; train={len(train_idx)} val={len(val_idx)}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    out = Path(config.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_artifact_loss = float("inf")
    history = []
    for epoch in range(1, config.epochs + 1):
        train_set.set_epoch(epoch)
        train_metrics = _run_epoch(model, train_loader, device, dtype, optimizer, scaler, config.hard_mix, config.value_weight)
        eval_loader = val_loader if val_loader is not None else train_loader
        with torch.no_grad():
            fp_metrics = _run_epoch(model, eval_loader, device, dtype, None, None, config.hard_mix, config.value_weight)
            artifact = quantized_snapshot(model)
            artifact_metrics = _run_epoch(artifact, eval_loader, device, None, None, None, config.hard_mix, config.value_weight)
        history.append({"epoch": epoch, "train": train_metrics, "val_fp": fp_metrics, "val_int8": artifact_metrics})
        print(f"epoch {epoch}: train loss {train_metrics['loss']:.4f} | val fp {fp_metrics['loss']:.4f} "
              f"| val int8 {artifact_metrics['loss']:.4f} agree {artifact_metrics['top_move_agreement']:.3f} "
              f"value_mae {artifact_metrics['value_mae']:.4f}")
        if artifact_metrics["loss"] < best_artifact_loss:
            best_artifact_loss = artifact_metrics["loss"]
            save_checkpoint(model, str(out), {"epoch": epoch, "val_int8": artifact_metrics, "val_fp": fp_metrics})
    Path(str(out) + ".history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"wrote {out} (best val int8 loss {best_artifact_loss:.4f})")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 9x9 student by distillation.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="checkpoints/student.pt")
    parser.add_argument("--init-checkpoint", help="warm start from a .pt checkpoint")
    parser.add_argument("--init-dir", help="warm start from a go-model.json/.bin directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hard-mix", type=float, default=0.5)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--non-deterministic", dest="deterministic", action="store_false")
    args = parser.parse_args()
    train(TrainConfig(**vars(args)))


if __name__ == "__main__":
    main()
