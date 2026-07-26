"""Generate the report's dependency-free SVG figures from fixed run evidence."""
from pathlib import Path

OUT = Path(__file__).parents[1] / "reports" / "hope-reproduction" / "images"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"HOPE": "#2563eb", "input-L1": "#ef4444", "joint-L1": "#f59e0b", "BN-scale": "#10b981"}


def frame(title, subtitle, body, width=760, height=450):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="55" y="38" font-family="system-ui,sans-serif" font-size="22" font-weight="700" fill="#111827">{title}</text>
<text x="55" y="62" font-family="system-ui,sans-serif" font-size="13" fill="#4b5563">{subtitle}</text>
{body}
</svg>"""


def line_chart(name, title, subtitle, densities, series, ymax):
    left, top, right, bottom = 72, 92, 725, 340
    x = lambda d: left + (1.0 - d) / (1.0 - min(densities)) * (right - left)
    y = lambda v: bottom - v / ymax * (bottom - top)
    lines = [
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#9ca3af"/>',
    ]
    for v in range(0, int(ymax) + 1, 10):
        yy = y(v)
        lines += [
            f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" stroke="#e5e7eb"/>',
            f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" font-family="system-ui" font-size="11" fill="#6b7280">{v}</text>',
        ]
    for d in densities:
        xx = x(d)
        lines.append(f'<text x="{xx:.1f}" y="{bottom+22}" text-anchor="middle" font-family="system-ui" font-size="11" fill="#6b7280">{int(d*100)}%</text>')
    lines += [
        f'<text x="18" y="{(top+bottom)/2}" transform="rotate(-90 18 {(top+bottom)/2})" text-anchor="middle" font-family="system-ui" font-size="12" fill="#374151">ImageNetV2 top-1 accuracy (%)</text>',
        f'<text x="{(left+right)/2}" y="426" text-anchor="middle" font-family="system-ui" font-size="12" fill="#374151">retained eligible channels</text>',
    ]
    for idx, (label, vals) in enumerate(series.items()):
        pts = " ".join(f"{x(d):.1f},{y(v):.1f}" for d, v in zip(densities, vals))
        color = COLORS[label]
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3"/>')
        for d, v in zip(densities, vals):
            lines.append(f'<circle cx="{x(d):.1f}" cy="{y(v):.1f}" r="4" fill="{color}"/>')
        lx = 95 + idx * 145
        lines += [
            f'<line x1="{lx}" y1="390" x2="{lx+24}" y2="390" stroke="{color}" stroke-width="3"/>',
            f'<text x="{lx+31}" y="394" font-family="system-ui" font-size="12" fill="#374151">{label}</text>',
        ]
    (OUT / name).write_text(frame(title, subtitle, "\n".join(lines)))


def bar_chart(name, title, subtitle, labels, values, xmax, x_label, percent=False):
    left, top, right, bottom = 180, 100, 710, 360
    row = (bottom - top) / len(labels)
    body = [f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#9ca3af"/>']
    for i in range(6):
        val = xmax * i / 5
        xx = left + (right-left) * i / 5
        tick = f"{val*100:.1f}%" if percent else f"{val:.1f}"
        body += [
            f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{bottom}" stroke="#e5e7eb"/>',
            f'<text x="{xx:.1f}" y="{bottom+20}" text-anchor="middle" font-family="system-ui" font-size="11" fill="#6b7280">{tick}</text>',
        ]
    for i, (label, value) in enumerate(zip(labels, values)):
        yy = top + i * row + 10
        color = COLORS.get(label, "#2563eb")
        width = value / xmax * (right-left)
        shown = f"{value*100:.1f}%" if percent else f"{value:.3f}"
        body += [
            f'<text x="{left-12}" y="{yy+19}" text-anchor="end" font-family="system-ui" font-size="13" fill="#374151">{label}</text>',
            f'<rect x="{left}" y="{yy}" width="{width:.1f}" height="28" rx="4" fill="{color}"/>',
            f'<text x="{left+width+7:.1f}" y="{yy+19}" font-family="system-ui" font-size="12" font-weight="700" fill="#111827">{shown}</text>',
        ]
    body.append(f'<text x="{(left+right)/2}" y="412" text-anchor="middle" font-family="system-ui" font-size="12" fill="#374151">{x_label}</text>')
    (OUT / name).write_text(frame(title, subtitle, "\n".join(body)))


line_chart(
    "resnet50_accuracy.svg",
    "HOPE retains ResNet-50 accuracy at dense pruning levels",
    "10,000 ImageNetV2 matched-frequency images; no fine-tuning or BN recalibration",
    [1.0, .95, .90, .85, .80],
    {"HOPE": [69.90, 66.51, 63.42, 41.47, 9.16], "input-L1": [69.90, .11, .14, .16, .18],
     "joint-L1": [69.90, 14.33, .36, .09, .12], "BN-scale": [69.90, 1.83, .78, .22, .18]},
    70,
)
line_chart(
    "resnet18_accuracy.svg",
    "The dense-regime advantage transfers to ResNet-18",
    "Interleaved checkpoints combine fresh seeds 71 and 113; same full ImageNetV2 benchmark",
    [1.0, .95, .90, .85, .80, .75],
    {"HOPE": [57.29, 52.03, 44.71, 34.42, 20.31, 5.39], "input-L1": [57.29, 17.92, .21, .12, .10, .20],
     "joint-L1": [57.29, 9.49, .31, .12, .16, .07], "BN-scale": [57.29, 36.92, 12.35, 3.24, .91, .43]},
    60,
)
bar_chart(
    "rescaling_invariance.svg",
    "HOPE alone preserves the pruning set under rescaling",
    "ResNet-50 exact function-preserving control; overlap of the bottom-ranked 25% before/after",
    ["HOPE", "input-L1", "joint-L1", "BN-scale"],
    [1.0, .5418538, .5225806, .3447293],
    1.0, "bottom-quartile Jaccard overlap", percent=False,
)
bar_chart(
    "kernel_validation.svg",
    "Closed-form ReLU kernel agrees with Monte Carlo estimates",
    "Median relative error over 64 sampled channels × 100,000 Gaussian draws",
    ["R50 seed 151", "R50 seed 17", "R50 seed 43", "R18 seed 113", "R18 seed 71"],
    [.0067264, .0063587, .0061252, .0083672, .0077713],
    .01, "median relative error", percent=True,
)
