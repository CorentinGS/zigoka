"""PyTorch replica of the Moka student (see vendor/moka/src/runtime.ts).

12 planes -> 32ch stem -> 12 nested-bottleneck blocks (32->16->16->32,
global mean/max adapters on blocks 4/8/12) -> policy head (4ch conv +
linear to 82) and value head (2ch conv + 32 hidden + tanh).
Input is channel-first [B, 12, 9, 9]; the vendored runtime consumes the
same values position-major, the layout conversion lives in manifest.py.
"""
from __future__ import annotations

import torch
from torch import nn

from . import AREA, INPUT_PLANES, MOVE_COUNT

TRUNK, BOTTLENECK, BLOCKS = 32, 16, 12
GLOBAL_INTERVAL, GLOBAL_HIDDEN = 4, 8
POLICY_CH, VALUE_CH, VALUE_HIDDEN = 4, 2, 32


class Block(nn.Module):
    def __init__(self, global_adapter: bool) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(TRUNK, BOTTLENECK, 1)
        self.first = nn.Conv2d(BOTTLENECK, BOTTLENECK, 3, padding=1)
        self.second = nn.Conv2d(BOTTLENECK, BOTTLENECK, 3, padding=1)
        self.expand = nn.Conv2d(BOTTLENECK, TRUNK, 1)
        self.global_hidden = nn.Linear(2 * BOTTLENECK, GLOBAL_HIDDEN) if global_adapter else None
        self.global_output = nn.Linear(GLOBAL_HIDDEN, BOTTLENECK) if global_adapter else None

    def forward(self, trunk: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.first(torch.relu(self.reduce(trunk))))
        if self.global_hidden is not None and self.global_output is not None:
            pooled = torch.cat([hidden.mean(dim=(2, 3)), hidden.amax(dim=(2, 3))], dim=1)
            bias = self.global_output(torch.relu(self.global_hidden(pooled)))
            hidden = hidden + bias[:, :, None, None]
        residual = self.expand(torch.relu(self.second(hidden)))
        return torch.relu(trunk + residual)


class MokaNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(INPUT_PLANES, TRUNK, 3, padding=1)
        self.blocks = nn.ModuleList(
            Block(global_adapter=(i + 1) % GLOBAL_INTERVAL == 0) for i in range(BLOCKS)
        )
        self.policy_convolution = nn.Conv2d(TRUNK, POLICY_CH, 1)
        self.policy_linear = nn.Linear(POLICY_CH * AREA, MOVE_COUNT)
        self.value_convolution = nn.Conv2d(TRUNK, VALUE_CH, 1)
        self.value_hidden = nn.Linear(VALUE_CH * AREA, VALUE_HIDDEN)
        self.value_output = nn.Linear(VALUE_HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trunk = torch.relu(self.stem(x))
        for block in self.blocks:
            trunk = block(trunk)
        # The vendored runtime flattens position-major (move * channels + ch),
        # so the linear heads must see that order, not torch's channel-major.
        policy = self.policy_linear(torch.relu(self.policy_convolution(trunk)).permute(0, 2, 3, 1).flatten(1))
        value_feats = torch.relu(self.value_convolution(trunk)).permute(0, 2, 3, 1).flatten(1)
        value = torch.tanh(self.value_output(torch.relu(self.value_hidden(value_feats)))).squeeze(1)
        return policy, value

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def manifest_items(self):
        """(manifest tensor name, torch tensor) pairs, owned by the model."""
        state = self.state_dict()
        for manifest_name, state_key in _MANIFEST_MAP:
            yield manifest_name, state[state_key]

    def load_manifest_items(self, items: dict) -> None:
        by_key = {state_key: items[manifest_name] for manifest_name, state_key in _MANIFEST_MAP}
        self.load_state_dict(by_key)


def _build_manifest_map() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = [("stem.weight", "stem.weight"), ("stem.bias", "stem.bias")]
    for i in range(BLOCKS):
        manifest_prefix, torch_prefix = f"residual.{i}", f"blocks.{i}"
        for suffix in ("reduce", "first"):
            pairs += [(f"{manifest_prefix}.{suffix}.weight", f"{torch_prefix}.{suffix}.weight"),
                      (f"{manifest_prefix}.{suffix}.bias", f"{torch_prefix}.{suffix}.bias")]
        if (i + 1) % GLOBAL_INTERVAL == 0:
            pairs += [(f"{manifest_prefix}.global.hidden.weight", f"{torch_prefix}.global_hidden.weight"),
                      (f"{manifest_prefix}.global.hidden.bias", f"{torch_prefix}.global_hidden.bias"),
                      (f"{manifest_prefix}.global.output.weight", f"{torch_prefix}.global_output.weight"),
                      (f"{manifest_prefix}.global.output.bias", f"{torch_prefix}.global_output.bias")]
        for suffix in ("second", "expand"):
            pairs += [(f"{manifest_prefix}.{suffix}.weight", f"{torch_prefix}.{suffix}.weight"),
                      (f"{manifest_prefix}.{suffix}.bias", f"{torch_prefix}.{suffix}.bias")]
    pairs += [
        ("policy.convolution.weight", "policy_convolution.weight"),
        ("policy.convolution.bias", "policy_convolution.bias"),
        ("policy.linear.weight", "policy_linear.weight"),
        ("policy.linear.bias", "policy_linear.bias"),
        ("value.convolution.weight", "value_convolution.weight"),
        ("value.convolution.bias", "value_convolution.bias"),
        ("value.hidden.weight", "value_hidden.weight"),
        ("value.hidden.bias", "value_hidden.bias"),
        ("value.output.weight", "value_output.weight"),
        ("value.output.bias", "value_output.bias"),
    ]
    return pairs


_MANIFEST_MAP = _build_manifest_map()


def save_checkpoint(model: MokaNet, path: str, meta: dict | None = None) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)


def load_checkpoint(path: str, device: torch.device) -> MokaNet:
    model = MokaNet()
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload)
    return model.to(device)
