"""go-generate: teacher-labelled positions from self-play / DAgger rollouts.

Teachers: a KataGo ONNX model (--katago, fresh distillation), a published
Moka artifact (--teacher-dir, bootstrap), or a .pt checkpoint (--teacher).
With --student, positions come from the student's own distribution while
the teacher labels every position (DAgger, upstream Moka's on-policy
recipe), optionally with teacher intervention on a fraction of moves.
Rollout temperature is applied exactly once, here at sampling time.
Whole games are kept intact so the trainer can split by game.
"""
from __future__ import annotations

import argparse
import torch
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import MAX_MOVES, MIN_PASS_MOVE, PASS_MOVE
from .board import GameState, encode_features, legal_moves, play_move
from .device import configure, describe, resolve_device
from .model import load_checkpoint
from .teachers import CheckpointTeacher, Teacher


@dataclass
class GenerateConfig:
    teacher: str | None = None
    teacher_dir: str | None = None
    katago: str | None = None
    student: str | None = None
    teacher_intervention: float = 0.15
    positions: int = 5000
    output: str = "data/distillation.npz"
    temperature: float = 1.0
    seed: int = 1
    device: str = "auto"


def _sample_move(probs: np.ndarray, state: GameState, rng: np.random.Generator, temperature: float) -> int:
    legal = legal_moves(state)
    choices = [m for m in legal if m != PASS_MOVE or state.move_count >= MIN_PASS_MOVE]
    weights = np.array([probs[m] for m in choices], dtype=np.float64)
    if temperature != 1.0:
        weights = np.power(np.maximum(weights, 1e-12), 1.0 / temperature)
    weights /= weights.sum()
    return int(rng.choice(choices, p=weights))


def _make_teacher(config: GenerateConfig, device: torch.device) -> Teacher:
    if config.katago:
        from .katago import KataGoTeacher

        return KataGoTeacher(config.katago, use_cuda=device.type == "cuda")
    if config.teacher_dir:
        return CheckpointTeacher.from_model_dir(config.teacher_dir, device)
    return CheckpointTeacher.from_checkpoint(config.teacher, device)


def generate(config: GenerateConfig) -> Path:
    device = resolve_device(config.device)
    configure(device)
    print(f"device: {describe(device)}")
    teacher = _make_teacher(config, device)
    student = load_checkpoint(config.student, device).eval() if config.student else None
    rng = np.random.default_rng(config.seed)

    features_l: list[np.ndarray] = []
    policy_l: list[np.ndarray] = []
    value_l: list[float] = []
    game_l: list[int] = []
    game_id = 0
    while len(value_l) < config.positions:
        state = GameState()
        game_feats: list[tuple[np.ndarray, np.ndarray, float]] = []
        while not state.is_game_over() and state.move_count < MAX_MOVES:
            probs, value = teacher.label_state(state)
            game_feats.append((encode_features(state), probs, value))
            actor_probs = probs
            if student is not None and rng.random() >= config.teacher_intervention:
                with torch.no_grad():
                    x = torch.from_numpy(encode_features(state)[None]).to(device)
                    logits, _ = student(x)
                    actor_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            move = _sample_move(actor_probs, state, rng, config.temperature)
            nxt = play_move(state, move)
            if nxt is None:
                break
            state = nxt
        for feats, probs, value in game_feats:
            features_l.append(feats)
            policy_l.append(probs)
            value_l.append(value)
            game_l.append(game_id)
        game_id += 1
    out = Path(config.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=np.array(features_l, dtype=np.float32),
                        policy=np.array(policy_l, dtype=np.float32),
                        value=np.array(value_l, dtype=np.float32),
                        game_id=np.array(game_l, dtype=np.int64))
    print(f"wrote {out}: {len(value_l)} positions from {game_id} games")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher-labelled 9x9 positions.")
    parser.add_argument("--teacher", help="teacher .pt checkpoint")
    parser.add_argument("--teacher-dir", help="teacher go-model.json/.bin directory (e.g. vendor/moka/model)")
    parser.add_argument("--katago", help="KataGo ONNX model path (fresh teacher distillation)")
    parser.add_argument("--student", help="student .pt checkpoint for DAgger rollouts")
    parser.add_argument("--teacher-intervention", type=float, default=0.15)
    parser.add_argument("--positions", type=int, default=5000)
    parser.add_argument("--output", default="data/distillation.npz")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    args = parser.parse_args()
    if not (args.teacher or args.teacher_dir or args.katago):
        parser.error("one of --teacher, --teacher-dir or --katago is required")
    generate(GenerateConfig(**vars(args)))


if __name__ == "__main__":
    main()
