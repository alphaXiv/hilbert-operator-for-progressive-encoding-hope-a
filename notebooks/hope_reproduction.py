import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Reproducing HOPE's rescaling and pruning claims

    **Verdict: partially reproduced.** Fresh Kubernetes experiments support both selected
    claims on public pretrained torchvision ResNet-50 and ResNet-18 models evaluated on all
    10,000 ImageNetV2 matched-frequency images. The protocol substitutes ImageNetV2 for the
    gated original ImageNet validation set and does not measure physical speedup.

    This notebook embeds the measured evidence, so opening it does not rerun the expensive
    experiment. The optional calculator below only evaluates the paper's closed-form kernel.
    """)
    return


@app.cell
def _():
    r50 = {
        "density": [100, 95, 90, 85, 80],
        "HOPE": [69.90, 66.51, 63.42, 41.47, 9.16],
        "input-L1": [69.90, 0.11, 0.14, 0.16, 0.18],
        "joint-L1": [69.90, 14.33, 0.36, 0.09, 0.12],
        "BN-scale": [69.90, 1.83, 0.78, 0.22, 0.18],
    }
    return (r50,)


@app.cell
def _(mo, r50):
    rows = []
    for i, density in enumerate(r50["density"]):
        rows.append({"density": f"{density}%", **{k: v[i] for k, v in r50.items() if k != "density"}})
    mo.vstack([
        mo.md("## Headline result\nTop-1 accuracy on the full public benchmark, without tuning or recalibration:"),
        mo.ui.table(rows, selection=None),
        mo.md(
            "**How to read this:** at 90% density, HOPE retained **63.42%** top-1 from a "
            "69.90% baseline; the best comparator retained **0.78%**. At 80%, even HOPE "
            "fell to 9.16%, so this evidence supports ordering quality—not robust extreme pruning."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What HOPE scores

    For a BatchNorm output modeled as \(z=\beta+|\gamma|X\), \(X\sim\mathcal N(0,1)\),
    the squared Hilbert norm after ReLU is

    \[
    K=(\gamma^2+\beta^2)\Phi(\beta/|\gamma|)
      +\beta|\gamma|\phi(\beta/|\gamma|).
    \]

    Multiplying \(\sqrt K\) by the outgoing weight norm gives neuron capacity. The
    distortion-rate score divides the paper's pruning penalty by the channel's static
    footprint. We implemented that score, input-weight L1, joint input/output L1, and
    BatchNorm-scale baselines over internal residual-block channels.
    """)
    return


@app.cell
def _(mo):
    beta = mo.ui.slider(-3.0, 3.0, value=0.0, step=0.1, label="β")
    gamma = mo.ui.slider(0.1, 3.0, value=1.0, step=0.1, label="|γ|")
    mo.hstack([beta, gamma], justify="start")
    return beta, gamma


@app.cell
def _(beta, gamma, mo):
    import math
    ratio = beta.value / gamma.value
    phi = math.exp(-ratio * ratio / 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + math.erf(ratio / math.sqrt(2)))
    kernel = (gamma.value**2 + beta.value**2) * Phi + beta.value * gamma.value * phi
    mo.md(f"Closed-form kernel: **{kernel:.6f}**; Hilbert norm: **{math.sqrt(max(kernel, 0)):.6f}**")
    return


@app.cell
def _(mo):
    invariance = [
        {"score": "HOPE", "bottom-25% overlap": 1.000, "Spearman": 1.000},
        {"score": "input-L1", "bottom-25% overlap": 0.542, "Spearman": 0.894},
        {"score": "joint-L1", "bottom-25% overlap": 0.523, "Spearman": 0.828},
        {"score": "BN-scale", "bottom-25% overlap": 0.345, "Spearman": 0.563},
    ]
    mo.vstack([
        mo.md("## Function-preserving rescaling"),
        mo.ui.table(invariance, selection=None),
        mo.md(
            "In the float64 ResNet-50 control, rescaling changed logits by at most "
            "**5.65×10⁻¹⁶ relative**. HOPE's complete order was identical and its maximum "
            "score change was 1.90×10⁻¹⁴; all magnitude-based pruning sets changed materially."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Numerical and scope checks

    Across five fresh model/seed conditions, the analytic kernel's median relative error
    against 100,000-sample Monte Carlo estimates was **0.61–0.84%** (64 channels each).
    The experiment used post-BatchNorm masks equivalent to deleting selected internal
    channels and downstream input slices. It used no fine-tuning, recalibration, merging,
    block eviction, or DEFT.

    Compute: Kubernetes, peak 16 concurrent **NVIDIA RTX PRO 6000 Blackwell** GPUs,
    0.42 hours measured experiment wall span. See the linked report and source repository
    for branch-level provenance and the exact fixed run command.
    """)
    return


if __name__ == "__main__":
    app.run()
