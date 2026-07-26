from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Channel:
    """One removable Conv-BN-ReLU channel and its downstream convolution."""

    layer: str
    index: int
    conv: nn.Conv2d
    bn: nn.BatchNorm2d
    outgoing: nn.Conv2d

    @property
    def key(self) -> str:
        return f"{self.layer}:{self.index}"


def relu_self_kernel(beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Equation (3): E[ReLU(Y)^2] for Y ~ Normal(beta, gamma^2)."""
    sigma = gamma.abs()
    deterministic = beta.clamp_min(0).square()
    safe_sigma = sigma.clamp_min(torch.finfo(beta.dtype).tiny)
    z = beta / safe_sigma
    cdf = torch.special.ndtr(z)
    pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    result = (gamma.square() + beta.square()) * cdf + beta * sigma * pdf
    return torch.where(sigma == 0, deterministic, result).clamp_min(0)


def collect_channels(model: nn.Module) -> list[Channel]:
    """Collect shape-safe internal channels from torchvision ResNet blocks."""
    channels: list[Channel] = []
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        stage = getattr(model, stage_name)
        for block_index, block in enumerate(stage):
            prefix = f"{stage_name}.{block_index}"
            channels.extend(
                Channel(prefix + ".bn1", i, block.conv1, block.bn1, block.conv2)
                for i in range(block.conv1.out_channels)
            )
            # Bottleneck blocks have a second internal channel between conv2/conv3.
            if hasattr(block, "conv3"):
                channels.extend(
                    Channel(prefix + ".bn2", i, block.conv2, block.bn2, block.conv3)
                    for i in range(block.conv2.out_channels)
                )
    return channels


def score_channels(channels: Iterable[Channel]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for channel in channels:
            i = channel.index
            incoming = channel.conv.weight[i].detach().float()
            outgoing = channel.outgoing.weight[:, i].detach().float()
            gamma = channel.bn.weight[i].detach().float()
            beta = channel.bn.bias[i].detach().float()
            kernel = relu_self_kernel(beta.reshape(()), gamma.reshape(()))
            capacity = outgoing.square().sum().sqrt() * kernel.sqrt()
            scores[channel.key] = {
                "capacity": float(capacity),
                "input_l1": float(incoming.abs().sum()),
                "joint_l1": float(incoming.abs().sum() + outgoing.abs().sum()),
                "bn_scale": float(gamma.abs()),
                "footprint": float(incoming.numel() + outgoing.numel() + 4),
            }
    return scores


def pruning_order(
    channels: list[Channel],
    scores: dict[str, dict[str, float]],
    method: str,
) -> list[str]:
    """Return a full global ordering, preserving at least one channel per layer."""
    by_layer: dict[str, list[str]] = {}
    for channel in channels:
        by_layer.setdefault(channel.layer, []).append(channel.key)

    if method != "hope":
        return sorted((c.key for c in channels), key=lambda key: (scores[key][method], key))

    active = {key for keys in by_layer.values() for key in keys}
    layer_energy = {
        layer: sum(scores[key]["capacity"] for key in keys)
        for layer, keys in by_layer.items()
    }
    layer_count = {layer: len(keys) for layer, keys in by_layer.items()}
    key_to_layer = {key: layer for layer, keys in by_layer.items() for key in keys}
    order: list[str] = []
    while active:
        best_key = None
        best_score = float("inf")
        for key in active:
            layer = key_to_layer[key]
            if layer_count[layer] <= 1:
                continue
            capacity = scores[key]["capacity"]
            residual = layer_energy[layer] - capacity
            if residual <= 1e-15:
                continue
            distortion = layer_count[layer] * capacity / residual
            rate = distortion / scores[key]["footprint"]
            candidate = (rate, key)
            if candidate < (best_score, best_key or ""):
                best_score, best_key = candidate
        if best_key is None:
            break
        active.remove(best_key)
        order.append(best_key)
        layer = key_to_layer[best_key]
        layer_energy[layer] -= scores[best_key]["capacity"]
        layer_count[layer] -= 1
    return order


def make_mask_map(
    channels: list[Channel], order: list[str], density: float, device: torch.device
) -> dict[str, torch.Tensor]:
    prune_count = int(round((1.0 - density) * len(channels)))
    pruned = set(order[:prune_count])
    masks: dict[str, torch.Tensor] = {}
    for channel in channels:
        if channel.layer not in masks:
            masks[channel.layer] = torch.ones(
                channel.bn.num_features, dtype=torch.float32, device=device
            )
        if channel.key in pruned:
            masks[channel.layer][channel.index] = 0.0
    return masks


def register_mask_hooks(
    channels: list[Channel], masks: dict[str, torch.Tensor]
) -> list[torch.utils.hooks.RemovableHandle]:
    handles = []
    seen: set[str] = set()
    for channel in channels:
        if channel.layer in seen:
            continue
        seen.add(channel.layer)
        mask = masks[channel.layer].reshape(1, -1, 1, 1)

        def hook(_module, _inputs, output, channel_mask=mask):
            return output * channel_mask.to(dtype=output.dtype)

        handles.append(channel.bn.register_forward_hook(hook))
    return handles


def rescale_model(
    model: nn.Module, seed: int
) -> tuple[nn.Module, dict[str, tuple[float, float]]]:
    """Compose BN-normalization and positive-homogeneity symmetries."""
    scaled = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(seed)
    channels = collect_channels(scaled)
    factors: dict[str, tuple[float, float]] = {}
    with torch.no_grad():
        for channel in channels:
            i = channel.index
            # s changes non-overlapping conv1 incoming magnitudes while BN cancels
            # it exactly. Scaling conv2 rows too would also change the output
            # geometry of the preceding eligible bn1 neurons.
            # t re-shards the post-BN ReLU signal into the downstream weights.
            s = (
                float(torch.exp(torch.empty(()).uniform_(-0.7, 0.7, generator=generator)))
                if channel.layer.endswith(".bn1")
                else 1.0
            )
            t = float(torch.exp(torch.empty(()).uniform_(-0.7, 0.7, generator=generator)))
            if s != 1.0:
                channel.conv.weight[i].mul_(s)
                channel.bn.running_mean[i].mul_(s)
                old_var = channel.bn.running_var[i].clone()
                channel.bn.running_var[i].copy_(
                    (old_var + channel.bn.eps) * s * s - channel.bn.eps
                )
            channel.bn.weight[i].mul_(t)
            channel.bn.bias[i].mul_(t)
            channel.outgoing.weight[:, i].div_(t)
            factors[channel.key] = (s, t)
    return scaled, factors


def rank_diagnostics(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
    channels_before: list[Channel],
    channels_after: list[Channel],
) -> dict[str, dict[str, float]]:
    diagnostics: dict[str, dict[str, float]] = {}
    for method in ("hope", "input_l1", "joint_l1", "bn_scale"):
        order_before = pruning_order(channels_before, before, method)
        order_after = pruning_order(channels_after, after, method)
        common = min(len(order_before), len(order_after))
        position_before = {key: i for i, key in enumerate(order_before[:common])}
        position_after = {key: i for i, key in enumerate(order_after[:common])}
        keys = sorted(set(position_before) & set(position_after))
        x = np.asarray([position_before[key] for key in keys], dtype=np.float64)
        y = np.asarray([position_after[key] for key in keys], dtype=np.float64)
        spearman = float(np.corrcoef(x, y)[0, 1]) if len(keys) > 1 else 1.0
        cutoff = max(1, common // 4)
        a, b = set(order_before[:cutoff]), set(order_after[:cutoff])
        jaccard = len(a & b) / len(a | b)
        exact = sum(a == b for a, b in zip(order_before[:common], order_after[:common])) / common
        if method == "hope":
            values_before = np.asarray([before[k]["capacity"] for k in before])
            values_after = np.asarray([after[k]["capacity"] for k in before])
        else:
            values_before = np.asarray([before[k][method] for k in before])
            values_after = np.asarray([after[k][method] for k in before])
        relative = np.abs(values_after - values_before) / np.maximum(np.abs(values_before), 1e-30)
        diagnostics[method] = {
            "spearman": spearman,
            "top25_jaccard": float(jaccard),
            "exact_position_fraction": float(exact),
            "median_score_relative_change": float(np.median(relative)),
            "max_score_relative_change": float(np.max(relative)),
        }
    return diagnostics


def monte_carlo_kernel_check(
    model: nn.Module, seed: int, samples: int, channels_to_test: int, device: torch.device
) -> dict[str, float]:
    channels = collect_channels(model)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(channels), generator=generator)[:channels_to_test].tolist()
    beta = torch.stack(
        [channels[i].bn.bias[channels[i].index].detach().float().cpu() for i in indices]
    ).to(device)
    gamma = torch.stack(
        [channels[i].bn.weight[channels[i].index].detach().float().cpu() for i in indices]
    ).to(device)
    analytic = relu_self_kernel(beta, gamma)
    gpu_generator = torch.Generator(device=device).manual_seed(seed + 1)
    noise = torch.randn((samples, len(indices)), generator=gpu_generator, device=device)
    empirical = torch.relu(beta.unsqueeze(0) + gamma.abs().unsqueeze(0) * noise).square().mean(0)
    relative = (empirical - analytic).abs() / analytic.clamp_min(1e-12)
    return {
        "channels": len(indices),
        "samples_per_channel": samples,
        "median_relative_error": float(relative.median()),
        "max_relative_error": float(relative.max()),
        "mean_relative_error": float(relative.mean()),
    }
