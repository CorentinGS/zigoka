"""go-train: distill teacher labels into the student.

Data: a legacy .npz, or compact shards (see dataset.load_training_data
for the targets they give). Every tensor goes to the GPU once, when it
fits (`--store auto`): a batch is a gather, the packed planes are
decoded and the dihedral augmentation applied on the GPU, one random
transform per position. 1M positions at 19×19 take about 1.6 GB.

Loss = (1-hard_mix) * soft CE + hard_mix * hard CE (Moka's sweep) +
value_weight * value MSE, plus two heads that only exist in training,
on the trunk: ownership (1×1 conv + tanh, MSE, `ownership_weight`)
and score lead (pooled trunk + linear, MSE on lead / score_scale,
`score_weight`). They shape the trunk and are not exported. Shards
without ownership or score skip them.

AdamW, linear warmup then cosine decay. Whole-game validation split.
Checkpoint selection scores the INT8-quantized snapshot (what the shipped
artifact computes) on policy + value, not the float weights.
Deterministic by default: shuffles and transforms come from one seeded
CPU generator.

Architecture: `--arch` (v1, v2) at the data's board size, or the
architecture of `--init-checkpoint` / `--init-dir`. A warm start from
another board size keeps the trunk and starts new heads (only the heads
depend on the board size).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataset import TrainingData, load_training_data, split_by_game, symmetry_permutations
from .device import amp_dtype, configure, describe, is_cuda, resolve_device
from .manifest import import_model, quantized_snapshot
from .model import Architecture, MokaNet, is_trunk_key, preset, read_checkpoint, save_checkpoint
from .shards import BINARY_PLANES, OWNERSHIP_SCALE


@dataclass
class TrainConfig:
    data: list[str] = field(default_factory=list)
    val_data: list[str] = field(default_factory=list)
    out: str = "checkpoints/student.pt"
    init_checkpoint: str | None = None
    init_dir: str | None = None
    arch: str = "v1"
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    warmup: float = 0.03
    min_lr: float = 0.05
    weight_decay: float = 0.0
    hard_mix: float = 0.5
    value_weight: float = 1.0
    visits_mix: float = 0.5
    outcome_mix: float = 0.3
    ownership_weight: float = 0.5
    score_weight: float = 0.1
    score_scale: float = 20.0
    val_fraction: float = 0.1
    seed: int = 1
    device: str = "auto"
    store: str = "auto"
    amp: bool = True
    deterministic: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.data, str):
            self.data = [self.data]
        if isinstance(self.val_data, str):
            self.val_data = [self.val_data]


class AuxHeads(nn.Module):
    """Training-only heads on the trunk: not part of the artifact."""

    def __init__(self, arch: Architecture) -> None:
        super().__init__()
        self.ownership = nn.Conv2d(arch.trunk, 1, 1)
        self.score = nn.Linear(2 * arch.trunk, 1)

    def forward(self, trunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ownership = torch.tanh(self.ownership(trunk)).flatten(1)
        pooled = torch.cat([trunk.mean(dim=(2, 3)), trunk.amax(dim=(2, 3))], dim=1)
        return ownership, self.score(pooled).squeeze(1)


class Store:
    """The training data as tensors on one device, batched by index."""

    def __init__(self, data: TrainingData, device: torch.device) -> None:
        self.size = data.size
        self.area = data.size * data.size
        self.device = device

        def put(array: np.ndarray | None) -> torch.Tensor | None:
            return None if array is None else torch.from_numpy(np.ascontiguousarray(array)).to(device)

        self.features = put(data.features)
        self.planes, self.komi = put(data.planes), put(data.komi)
        self.policy, self.value = put(data.policy), put(data.value)
        self.ownership, self.has_ownership = put(data.ownership), put(data.has_ownership)
        self.score, self.has_score = put(data.score), put(data.has_score)
        self.has_aux = data.ownership is not None and bool(data.has_ownership.any() or data.has_score.any())
        self.shifts = torch.arange(7, -1, -1, device=device, dtype=torch.uint8)

    @staticmethod
    def nbytes(data: TrainingData) -> int:
        arrays = (data.features, data.planes, data.komi, data.policy, data.value, data.ownership,
                  data.has_ownership, data.score, data.has_score)
        return sum(a.nbytes for a in arrays if a is not None)

    def batch(self, index: torch.Tensor, target: torch.device) -> dict[str, torch.Tensor]:
        index = index.to(self.device)
        out = {"policy": self.policy[index], "value": self.value[index]}
        if self.features is not None:
            out["features"] = self.features[index]
        else:
            out["planes"], out["komi"] = self.planes[index], self.komi[index]
        if self.has_aux:
            for name in ("ownership", "has_ownership", "score", "has_score"):
                out[name] = getattr(self, name)[index]
        out = {k: v.to(target, non_blocking=True) for k, v in out.items()}
        if "planes" in out:
            out["features"] = decode_planes(out.pop("planes"), out.pop("komi"), self.size, self.shifts.to(target))
        out["policy"] = out["policy"].float()
        if "ownership" in out:
            out["ownership"] = out["ownership"].float() / OWNERSHIP_SCALE
        return out


def decode_planes(planes: torch.Tensor, komi: torch.Tensor, size: int, shifts: torch.Tensor) -> torch.Tensor:
    """[B, P] packed planes 0-10 (np.packbits order) + [B] komi -> [B, 12, S, S] f32."""
    batch, area = len(planes), size * size
    bits = (planes.unsqueeze(-1) >> shifts) & 1
    binary = bits.reshape(batch, -1)[:, : BINARY_PLANES * area].reshape(batch, BINARY_PLANES, size, size).float()
    return torch.cat([binary, komi.float()[:, None, None, None].expand(batch, 1, size, size)], dim=1)


def augment(batch: dict[str, torch.Tensor], perms: torch.Tensor, generator: torch.Generator) -> None:
    """One random dihedral transform per position, in place."""
    features = batch["features"]
    count, channels, size = features.shape[0], features.shape[1], features.shape[-1]
    area = size * size
    perm = perms[torch.randint(0, len(perms), (count,), generator=generator).to(perms.device)]
    batch["features"] = features.reshape(count, channels, area).gather(
        2, perm[:, None, :].expand(count, channels, area)).reshape(count, channels, size, size)
    policy = batch["policy"]
    batch["policy"] = torch.cat([policy[:, :area].gather(1, perm), policy[:, area:]], dim=1)
    if "ownership" in batch:
        batch["ownership"] = batch["ownership"].gather(1, perm)


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = (prediction.float() - target) ** 2
    if error.dim() > 1:
        error = error.mean(dim=1)
    mask = mask.float()
    return (error * mask).sum() / mask.sum().clamp(min=1.0)


def compute_losses(model: MokaNet, aux: AuxHeads | None, batch: dict[str, torch.Tensor],
                   config: TrainConfig) -> dict[str, torch.Tensor]:
    trunk = model.trunk(batch["features"])
    logits, value = model.heads(trunk)
    log_probs = torch.log_softmax(logits.float(), dim=1)
    policy = batch["policy"]
    soft = -(policy * log_probs).sum(1).mean()
    hard = nn.functional.nll_loss(log_probs, policy.argmax(1))
    policy_loss = (1 - config.hard_mix) * soft + config.hard_mix * hard
    value_loss = nn.functional.mse_loss(value.float(), batch["value"])
    # The soft CE minus the target's entropy is the KL to the teacher.
    entropy = -(policy * torch.log(policy.clamp(min=1e-12))).sum(1).mean()
    out = {"policy": policy_loss, "value": value_loss, "main": policy_loss + config.value_weight * value_loss,
           "soft": soft, "target_entropy": entropy,
           "agree": (logits.argmax(1) == policy.argmax(1)).float().mean(),
           "value_mae": (value.float() - batch["value"]).abs().mean()}
    total = out["main"]
    if aux is not None and "ownership" in batch:
        ownership, score = aux(trunk)
        out["ownership"] = _masked_mse(ownership, batch["ownership"], batch["has_ownership"])
        out["score"] = _masked_mse(score, batch["score"], batch["has_score"])
        total = total + config.ownership_weight * out["ownership"] + config.score_weight * out["score"]
    out["total"] = total
    return out


def _evaluate(model: MokaNet, aux: AuxHeads | None, store: Store, indices: np.ndarray, device: torch.device,
              dtype: torch.dtype | None, config: TrainConfig) -> dict[str, float]:
    model.eval()
    if aux is not None:
        aux.eval()
    sums: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for start in range(0, len(indices), config.batch_size):
            index = torch.from_numpy(indices[start:start + config.batch_size])
            batch = store.batch(index, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                losses = compute_losses(model, aux, batch, config)
            for name, loss in losses.items():
                sums[name] = sums.get(name, 0) + loss.detach().float() * len(index)
    return {name: float(total) / len(indices) for name, total in sums.items()}


def _build_model(config: TrainConfig, size: int, device: torch.device) -> tuple[MokaNet, AuxHeads, str]:
    source, aux_state = None, None
    if config.init_checkpoint:
        payload = read_checkpoint(config.init_checkpoint, torch.device("cpu"))
        arch = Architecture(**payload["architecture"]) if "architecture" in payload else preset("v1")
        source = MokaNet(arch)
        source.load_state_dict(payload["state_dict"])
        aux_state = payload.get("aux_state_dict")
    elif config.init_dir:
        source = import_model(config.init_dir)
    if source is None:
        model = MokaNet(preset(config.arch, size))
        origin = f"fresh {config.arch}"
    elif source.arch.board_size == size:
        model, origin = source, "warm start"
    else:
        model = MokaNet(source.arch.with_board_size(size))
        trunk = {k: v for k, v in source.state_dict().items() if is_trunk_key(k)}
        missing, unexpected = model.load_state_dict(trunk, strict=False)
        assert not unexpected and all(not is_trunk_key(k) for k in missing)
        origin = f"trunk from {source.arch.board_size}×{source.arch.board_size}, new heads"
        aux_state = None
    aux = AuxHeads(model.arch)
    if aux_state is not None:
        aux.load_state_dict(aux_state)
    return model.to(device), aux.to(device), origin


def _schedule(config: TrainConfig, total_steps: int):
    warmup = max(1, int(config.warmup * total_steps))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return config.min_lr + (1 - config.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def _store_device(config: TrainConfig, data: TrainingData, device: torch.device) -> torch.device:
    if config.store == "cpu" or not is_cuda(device):
        return torch.device("cpu")
    if config.store == "device":
        return device
    free, _ = torch.cuda.mem_get_info(device)
    return device if Store.nbytes(data) < 0.5 * free else torch.device("cpu")


def train(config: TrainConfig) -> Path:
    device = resolve_device(config.device)
    configure(device, deterministic=config.deterministic)
    dtype = amp_dtype(device) if config.amp else None
    print(f"device: {describe(device)}; amp: {dtype}; deterministic: {config.deterministic}")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    data = load_training_data(config.data, config.visits_mix, config.outcome_mix, config.score_scale)
    store = Store(data, _store_device(config, data, device))
    if config.val_data:
        # A fixed validation set: the whole of `data` trains.
        val_data = load_training_data(config.val_data, config.visits_mix, config.outcome_mix, config.score_scale)
        if val_data.size != data.size:
            raise ValueError(f"validation data is {val_data.size}×{val_data.size}, training data {data.size}×{data.size}")
        val_store = Store(val_data, _store_device(config, val_data, device))
        train_idx, val_idx = np.arange(len(data)), np.arange(len(val_data))
    else:
        train_idx, val_idx = split_by_game({"game_id": data.game_id}, config.val_fraction, config.seed)
        val_store = store
    perms = symmetry_permutations(data.size).to(device)
    model, aux, origin = _build_model(config, data.size, device)
    if not store.has_aux:
        aux = None
    print(f"student {model.arch.board_size}×{model.arch.board_size}, {model.parameter_count():,} parameters "
          f"({origin}); train={len(train_idx)} val={len(val_idx)}; store on {store.device} "
          f"({Store.nbytes(data) / 2**20:.0f} MiB); aux heads: {aux is not None}")

    params = list(model.parameters()) + (list(aux.parameters()) if aux is not None else [])
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(train_idx) / config.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _schedule(config, config.epochs * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    out = Path(config.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    eval_store, eval_idx = (val_store, val_idx) if len(val_idx) else (store, train_idx)
    best_artifact_loss = float("inf")
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        if aux is not None:
            aux.train()
        started = time.monotonic()
        order = train_idx[torch.randperm(len(train_idx), generator=generator).numpy()]
        sums: dict[str, torch.Tensor] = {}
        for start in range(0, len(order), config.batch_size):
            index = torch.from_numpy(order[start:start + config.batch_size])
            batch = store.batch(index, device)
            augment(batch, perms, generator)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                losses = compute_losses(model, aux, batch, config)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            for name, loss in losses.items():
                sums[name] = sums.get(name, 0) + loss.detach().float() * len(index)
        train_metrics = {name: float(total) / len(order) for name, total in sums.items()}
        if not math.isfinite(train_metrics["total"]):
            raise FloatingPointError(f"epoch {epoch}: training loss is {train_metrics['total']}")
        seconds = time.monotonic() - started
        fp_metrics = _evaluate(model, aux, eval_store, eval_idx, device, dtype, config)
        artifact_metrics = _evaluate(quantized_snapshot(model), None, eval_store, eval_idx, device, None, config)
        history.append({"epoch": epoch, "lr": scheduler.get_last_lr()[0], "seconds": seconds,
                        "train": train_metrics, "val_fp": fp_metrics, "val_int8": artifact_metrics})
        print(f"epoch {epoch}: {len(order) / seconds:.0f} pos/s | train {train_metrics['total']:.4f} "
              f"| val fp {fp_metrics['main']:.4f} | val int8 {artifact_metrics['main']:.4f} "
              f"(policy {artifact_metrics['policy']:.4f}, KL {artifact_metrics['soft'] - artifact_metrics['target_entropy']:.4f}, "
              f"value {artifact_metrics['value']:.4f}) "
              f"agree {artifact_metrics['agree']:.3f} value_mae {artifact_metrics['value_mae']:.4f}", flush=True)
        if artifact_metrics["main"] < best_artifact_loss:
            best_artifact_loss = artifact_metrics["main"]
            save_checkpoint(model, str(out), {"epoch": epoch, "val_int8": artifact_metrics, "val_fp": fp_metrics},
                            extra={"aux_state_dict": aux.state_dict() if aux is not None else None,
                                   "config": asdict(config)})
    Path(str(out) + ".history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"wrote {out} (best val int8 loss {best_artifact_loss:.4f})")
    return out


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train the student by distillation.")
    parser.add_argument("--data", nargs="+", required=True, help="one .npz, or shard directories / globs")
    parser.add_argument("--val-data", nargs="+", default=[],
                        help="a fixed validation set (then --val-fraction is ignored and all of --data trains)")
    parser.add_argument("--out", default=defaults.out)
    parser.add_argument("--init-checkpoint", help="warm start from a .pt checkpoint")
    parser.add_argument("--init-dir", help="warm start from a go-model.json/.bin directory")
    parser.add_argument("--arch", default=defaults.arch, choices=("v1", "v2"), help="without a warm start")
    for name in ("epochs", "batch_size", "seed"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    for name in ("lr", "warmup", "min_lr", "weight_decay", "hard_mix", "value_weight", "visits_mix", "outcome_mix",
                 "ownership_weight", "score_weight", "score_scale", "val_fraction"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    parser.add_argument("--store", default="auto", choices=("auto", "device", "cpu"),
                        help="where the training tensors live (auto: the GPU when they fit)")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--non-deterministic", dest="deterministic", action="store_false")
    train(TrainConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
