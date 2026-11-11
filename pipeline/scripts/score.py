"""Scores checkpoints on a held-out shard directory, as docs/designs/moka-19.md reports them.

Each checkpoint is scored through its INT8 snapshot, the artifact the trainer selects and go-export
writes: the policy's KL to the teacher and its top-1 agreement, then the value against the teacher's
value alone (MSE, correlation, sign). The first line gives the MSE of always predicting 0.

usage (from pipeline/): .venv/bin/python scripts/score.py SHARD_DIR CHECKPOINT...
"""
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from go_pipeline.dataset import load_training_data
from go_pipeline.device import configure, resolve_device
from go_pipeline.manifest import quantized_snapshot
from go_pipeline.model import load_checkpoint
from go_pipeline.train import Store, compute_losses

BATCH = 512


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    paths = sorted(str(p) for p in root.glob("shard-*"))
    games = [json.loads(line) for p in paths for line in open(Path(p) / "games.jsonl")]
    device = resolve_device("auto")
    configure(device, deterministic=False)
    data = load_training_data(paths, outcome_mix=0.0)
    store = Store(data, device)
    teacher = data.value.astype(np.float64)
    ends = {end: sum(g["end"] == end for g in games) for end in sorted({g["end"] for g in games})}
    print(f"{root}: {len(teacher)} positions, {len(games)} games ({ends}), mean length "
          f"{statistics.mean(len(g['moves']) for g in games):.0f} moves; "
          f"value MSE of predicting 0 = {(teacher ** 2).mean():.4f}")
    config = SimpleNamespace(hard_mix=0.5, value_weight=1.0)
    for path in sys.argv[2:]:
        model = quantized_snapshot(load_checkpoint(path, device)).eval()
        sums: dict[str, float] = {}
        values = []
        with torch.no_grad():
            for start in range(0, len(teacher), BATCH):
                index = torch.arange(start, min(len(teacher), start + BATCH))
                batch = store.batch(index, device)
                for key, loss in compute_losses(model, None, batch, config).items():
                    sums[key] = sums.get(key, 0.0) + float(loss) * len(index)
                values.append(model(batch["features"])[1].float().squeeze(-1).cpu().numpy())
        means = {key: total / len(teacher) for key, total in sums.items()}
        predicted = np.concatenate(values).astype(np.float64)
        print(f"{Path(path).stem}: KL {means['soft'] - means['target_entropy']:.3f} "
              f"(target H {means['target_entropy']:.3f}) agree {means['agree']:.3f} | "
              f"value MSE {((predicted - teacher) ** 2).mean():.4f} "
              f"corr {np.corrcoef(predicted, teacher)[0, 1]:.3f} "
              f"sign {(np.sign(predicted) == np.sign(teacher)).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
