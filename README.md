# HOPE claim reproduction

This public repository reproduces two claims from **Hilbert Operator for Progressive Encoding (HOPE), arXiv:2607.21366**: its function-space channel score should be invariant under function-preserving positive rescaling, and its channel ordering should preserve more validation accuracy than input-L1, joint-L1, or BatchNorm-scale pruning.

**Assessment: partially reproduced.** On the full 10,000-image ImageNetV2 matched-frequency benchmark, pretrained ResNet-50 retained **63.42% top-1 at 90% channel density with HOPE**, versus **0.78% for the best baseline**, from a 69.90% unpruned baseline. The paper reports the comparison as a qualitative curve without numeric coordinates. A float64 symmetry control preserved logits to \(5.65\times10^{-16}\) relative error and HOPE's complete ordering exactly; magnitude pruning-set overlap fell to 0.345–0.542.

The reproduction substitutes public torchvision ResNet-50/18 and ImageNetV2 for the paper's Keras ResNet-50 and gated ImageNet validation set. It masks internal block channels without BatchNorm recalibration, fine-tuning, merging, block eviction, DEFT, or physical speedup measurement. Compute was Kubernetes with a peak of 16 concurrent **NVIDIA RTX PRO 6000 Blackwell** GPUs and a measured experiment wall span of **0.42 hours**.

- [Tutorial-style detailed report](reports/hope-reproduction/report.md)
- [Self-contained marimo notebook](notebooks/hope_reproduction.py)
- [Machine-readable results](reports/hope-reproduction/results.json)
- [Original paper](https://arxiv.org/abs/2607.21366)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/hilbert-operator-for-progressive-encoding-hope-a/blob/main/notebooks/hope_reproduction.py)

## Experiment log

The fixed command below is copied verbatim from `orx exp status`; each listed experiment ran on four Kubernetes GPUs.

| Branch / experiment | Purpose | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Public implementation and publication surface | Not run as an experiment (publication surface) | Report, figures, notebook, and metadata | — |
| [ResNet-50 primary](https://github.com/alphaXiv/hilbert-operator-for-progressive-encoding-hope-a/tree/orx/isolated-rescaling-and-shared-memory) | 100–50% densities; corrected isolated rescaling | `python -m pip install --disable-pip-version-check -r requirements.txt && python -m hope_repro.run` | Aligned at dense densities; 63.42% HOPE at 90% | 4× RTX PRO 6000 Blackwell |
| [ResNet-50 dense](https://github.com/alphaXiv/hilbert-operator-for-progressive-encoding-hope-a/tree/orx/isolated-resnet-50-dense) | 95/90/85/80% checkpoints | `python -m pip install --disable-pip-version-check -r requirements.txt && python -m hope_repro.run` | HOPE led every matched checkpoint | 4× RTX PRO 6000 Blackwell |
| [ResNet-50 sparse](https://github.com/alphaXiv/hilbert-operator-for-progressive-encoding-hope-a/tree/orx/isolated-resnet-50-sparse) | 75/65/55/45% diagnostic | `python -m pip install --disable-pip-version-check -r requirements.txt && python -m hope_repro.run` | HOPE best, but all methods lost useful accuracy | 4× RTX PRO 6000 Blackwell |
| [ResNet-50 symmetry control](https://github.com/alphaXiv/hilbert-operator-for-progressive-encoding-hope-a/tree/orx/resnet-50-nonadjacent-symmetry-control) | Float64 function and ordering invariance | `python -m pip install --disable-pip-version-check -r requirements.txt && python -m hope_repro.run` | Exact HOPE order; magnitude sets changed | 4× RTX PRO 6000 Blackwell |
| [ResNet-18 robustness](https://github.com/alphaXiv/hilbert-operator-for-progressive-encoding-hope-a/tree/orx/isolated-resnet-18-robustness) | Architecture control | `python -m pip install --disable-pip-version-check -r requirements.txt && python -m hope_repro.run` | Dense-regime advantage aligned | 4× RTX PRO 6000 Blackwell |
| [ResNet-18 interleaved seed](https://github.com/alphaXiv/hilbert-operator-for-progressive-encoding-hope-a/tree/orx/resnet-18-interleaved-density-seed-113) | Second seed and intermediate densities | `python -m pip install --disable-pip-version-check -r requirements.txt && python -m hope_repro.run` | Confirmed smooth dense-regime trend | 4× RTX PRO 6000 Blackwell |

## Reproduce

The committed default configuration evaluates dense ResNet-50 checkpoints. The formal OpenResearch invocation is:

```bash
orx exp run <experiment-id> --backend k8s
```

The immutable experiment command is:

```bash
python -m pip install --disable-pip-version-check -r requirements.txt && python -m hope_repro.run
```

It downloads public pretrained torchvision weights and the public ImageNetV2 matched-frequency archive, then prints configuration, compute identity, mechanism diagnostics, every accuracy point, and one terminal `FINAL_RESULT` JSON record. Change scientific conditions in committed `experiment.json`, not in the command.

For a local tutorial view:

```bash
marimo edit notebooks/hope_reproduction.py
marimo run notebooks/hope_reproduction.py
```

## Implementation map

- `hope_repro/core.py` implements the closed-form ReLU Hilbert kernel, capacity, distortion-rate score, baselines, channel collection, rescaling, masking, and Monte Carlo validation.
- `hope_repro/run.py` downloads ImageNetV2, launches one evaluation worker per pruning method, and emits complete log evidence.
- `.orx/k8s.yaml` requests four Blackwell GPUs with CUDA 12.8 and shared memory.
- `experiment.json` is the committed scientific configuration.

No model weights or dataset content are redistributed by this repository.
