from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models

from .core import (
    collect_channels,
    make_mask_map,
    monte_carlo_kernel_check,
    pruning_order,
    rank_diagnostics,
    register_mask_hooks,
    rescale_model,
    score_channels,
)


DATA_URL = (
    "https://s3-us-west-2.amazonaws.com/imagenetv2public/"
    "imagenetv2-matched-frequency.tar.gz"
)
METHODS = ("hope", "input_l1", "joint_l1", "bn_scale")


class NumericImageFolder(Dataset):
    """ImageNetV2 directory reader that treats folder names as integer labels."""

    def __init__(self, root: Path, transform):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for class_dir in sorted(root.iterdir(), key=lambda path: int(path.name)):
            if not class_dir.is_dir():
                continue
            label = int(class_dir.name)
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    self.samples.append((image_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label


def config() -> dict:
    defaults = {
        "condition": "resnet50-primary",
        "model": "resnet50",
        "seed": 17,
        "subset_size": 10000,
        "densities": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        "batch_size": 160,
        "workers_per_gpu": 6,
        "mc_samples": 100000,
        "mc_channels": 64,
    }
    path = Path("experiment.json")
    if path.exists():
        defaults.update(json.loads(path.read_text()))
    return defaults


def prepare_data() -> Path:
    base = Path(os.environ.get("HOPE_DATA_DIR", "/tmp/hope-data"))
    root = base / "imagenetv2-matched-frequency-format-val"
    if root.exists() and sum(1 for _ in root.glob("*/*")) >= 10000:
        return root
    base.mkdir(parents=True, exist_ok=True)
    archive = base / "imagenetv2-matched-frequency.tar.gz"
    if not archive.exists():
        print(f"DATA_DOWNLOAD {DATA_URL}", flush=True)
        urllib.request.urlretrieve(DATA_URL, archive)
    print(f"DATA_EXTRACT {archive}", flush=True)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(base, filter="data")
    if not root.exists():
        candidates = list(base.glob("imagenetv2-*format-val"))
        if not candidates:
            raise FileNotFoundError("ImageNetV2 extraction did not produce the expected directory")
        root = candidates[0]
    return root


def model_and_weights(model_name: str):
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        constructor = models.resnet50
    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        constructor = models.resnet18
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return constructor(weights=weights), weights


@torch.inference_mode()
def evaluate(model, loader, device) -> dict[str, float]:
    correct1 = 0
    correct5 = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        predictions = logits.topk(5, dim=1).indices
        correct = predictions.eq(labels.unsqueeze(1))
        correct1 += int(correct[:, :1].sum())
        correct5 += int(correct.sum())
        count += labels.numel()
    return {
        "top1": 100.0 * correct1 / count,
        "top5": 100.0 * correct5 / count,
        "images": count,
    }


def method_worker(method: str, gpu_index: int, cfg: dict, data_root: str) -> dict:
    torch.cuda.set_device(gpu_index)
    device = torch.device("cuda", gpu_index)
    model, weights = model_and_weights(cfg["model"])
    model.eval().to(device)
    channels = collect_channels(model)
    scores = score_channels(channels)
    order = pruning_order(channels, scores, method)
    dataset: Dataset = NumericImageFolder(Path(data_root), weights.transforms())
    if cfg["subset_size"] < len(dataset):
        generator = torch.Generator().manual_seed(cfg["seed"])
        indices = torch.randperm(len(dataset), generator=generator)[: cfg["subset_size"]].tolist()
        dataset = Subset(dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["workers_per_gpu"],
        pin_memory=True,
        persistent_workers=cfg["workers_per_gpu"] > 0,
    )
    points = []
    started = time.monotonic()
    for density in sorted(set(cfg["densities"]), reverse=True):
        masks = make_mask_map(channels, order, density, device)
        handles = register_mask_hooks(channels, masks)
        metrics = evaluate(model, loader, device)
        for handle in handles:
            handle.remove()
        metrics["target_density"] = density
        metrics["actual_density"] = 1.0 - int(round((1.0 - density) * len(channels))) / len(channels)
        metrics["pruned_channels"] = int(round((1.0 - density) * len(channels)))
        points.append(metrics)
        print(
            "POINT "
            + json.dumps(
                {"condition": cfg["condition"], "method": method, **metrics},
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        "method": method,
        "eligible_channels": len(channels),
        "points": points,
        "elapsed_seconds": time.monotonic() - started,
    }


@torch.inference_mode()
def mechanism_checks(cfg: dict) -> dict:
    device = torch.device("cuda", 0)
    model, _ = model_and_weights(cfg["model"])
    model.eval()
    before_channels = collect_channels(model)
    before_scores = score_channels(before_channels)
    scaled, factors = rescale_model(model, cfg["seed"] + 1000)
    scaled.eval()
    after_channels = collect_channels(scaled)
    after_scores = score_channels(after_channels)
    diagnostics = rank_diagnostics(before_scores, after_scores, before_channels, after_channels)

    model.to(device)
    scaled.to(device)
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 2000)
    probe = torch.randn((4, 3, 224, 224), generator=generator, device=device)
    original_logits = model(probe)
    scaled_logits = scaled(probe)
    absolute = (original_logits - scaled_logits).abs()
    function_check = {
        "max_abs_logit_difference": float(absolute.max()),
        "mean_abs_logit_difference": float(absolute.mean()),
        "max_relative_to_logit": float(
            absolute.max() / original_logits.abs().max().clamp_min(1e-12)
        ),
        "rescaled_channels": len(factors),
        "s_range": [min(x[0] for x in factors.values()), max(x[0] for x in factors.values())],
        "t_range": [min(x[1] for x in factors.values()), max(x[1] for x in factors.values())],
    }
    mc = monte_carlo_kernel_check(
        model,
        cfg["seed"] + 3000,
        cfg["mc_samples"],
        cfg["mc_channels"],
        device,
    )
    return {
        "function_preservation": function_check,
        "ranking": diagnostics,
        "monte_carlo": mc,
    }


def main() -> None:
    started_wall = time.time()
    started = time.monotonic()
    cfg = config()
    print("CONFIG " + json.dumps(cfg, sort_keys=True), flush=True)
    print(
        "COMPUTE "
        + json.dumps(
            {
                "backend": "kubernetes",
                "gpu_count": torch.cuda.device_count(),
                "gpu_models": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "torch": torch.__version__,
                "torchvision_model": cfg["model"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if torch.cuda.device_count() < 4:
        raise RuntimeError("The committed experiment contract requires four visible GPUs")

    root = prepare_data()
    # Populate the shared torch cache before spawning four workers.
    warm_model, _ = model_and_weights(cfg["model"])
    del warm_model
    checks = mechanism_checks(cfg)
    print("MECHANISM " + json.dumps(checks, sort_keys=True), flush=True)

    context = multiprocessing.get_context("spawn")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        futures = [
            pool.submit(method_worker, method, gpu, cfg, str(root))
            for gpu, method in enumerate(METHODS)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda result: METHODS.index(result["method"]))
    summary = {
        "schema": 1,
        "fresh_evidence_not_before": "2026-07-26T02:02:01.267Z",
        "condition": cfg["condition"],
        "config": cfg,
        "compute": {
            "backend": "kubernetes",
            "gpu_model": torch.cuda.get_device_name(0),
            "gpu_count": 4,
        },
        "dataset": {
            "name": "ImageNetV2 matched-frequency",
            "url": DATA_URL,
            "images": cfg["subset_size"],
            "substitution": "Public ImageNetV2 benchmark replaces the gated original ImageNet validation set.",
        },
        "scope": {
            "eligible": "Internal ResNet block channels (Bottleneck conv1/conv2; BasicBlock conv1)",
            "physical_realization": "Post-BN channel masks, functionally equivalent to deleting each selected internal channel and its downstream input slice.",
            "recalibration": False,
            "fine_tuning": False,
            "merging": False,
            "block_eviction": False,
            "deft": False,
        },
        "mechanism": checks,
        "methods": results,
        "started_unix": started_wall,
        "elapsed_seconds": time.monotonic() - started,
    }
    print("FINAL_RESULT " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

