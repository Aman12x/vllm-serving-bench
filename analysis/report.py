"""Build the tables and charts from results/raw/*.json.

    python analysis/report.py

Writes analysis/figures/{tradeoff,peruser}-{light,dark}.png and prints Markdown tables
(throughput/latency, accuracy, cost) for the README. Every number comes from the JSON
files; nothing here is hand-entered except GPU_USD_PER_HOUR.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"
FIG = ROOT / "analysis/figures"

# Published on-demand rate for one NVIDIA L4, used to turn tokens/s into $/1M tokens.
# Source: Modal's published per-second L4 price ($0.000222/s = $0.80/h), read 19 Sep 2026.
GPU_USD_PER_HOUR = 0.80

ORDER = ["bf16", "fp8", "awq"]
NICE = {"bf16": "BF16 (16-bit)", "fp8": "FP8 (8-bit)", "awq": "AWQ (4-bit)"}
LIGHT = {"bf16": "#2a78d6", "fp8": "#eb6834", "awq": "#1baf7a"}
DARK = {"bf16": "#3987e5", "fp8": "#d95926", "awq": "#199e70"}
THEME = {
    "light": dict(colors=LIGHT, surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df"),
    "dark": dict(colors=DARK, surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#3a3a38"),
}


def load_runs():
    runs = {}
    for f in sorted(RAW.glob("*_c*.json")):
        label = f.name.split("_c")[0]
        d = json.loads(f.read_text())
        runs.setdefault(label, {})[d["max_concurrency"]] = d
    return {k: dict(sorted(v.items())) for k, v in runs.items() if k in ORDER}


def cost_per_1m(throughput):
    return GPU_USD_PER_HOUR / 3600 / throughput * 1e6


def line_chart(runs, mode, x_key, y_key, xlabel, ylabel, title, fname, ylog=False):
    t = THEME[mode]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    for label in ORDER:
        pts = runs[label]
        xs = [pts[c][x_key] for c in pts]
        ys = [pts[c][y_key] for c in pts]
        color = t["colors"][label]
        ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=6,
                markeredgecolor=t["surface"], markeredgewidth=2, label=NICE[label], zorder=3)
        # direct label at the high-load end, so identity is never colour-alone
        ax.annotate(NICE[label], (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 6),
                    color=t["ink"], fontsize=9, fontweight="bold")
        for c, x, y in zip(pts, xs, ys):          # concurrency tags on each point
            ax.annotate(str(c), (x, y), textcoords="offset points", xytext=(0, -13),
                        ha="center", color=t["ink2"], fontsize=7.5)

    if ylog:
        ax.set_yscale("log")
    ax.margins(x=0.16, y=0.12)   # headroom so the end labels do not run off the canvas
    ax.set_xlabel(xlabel, color=t["ink2"], fontsize=9.5)
    ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9.5)
    ax.set_title(title, color=t["ink"], fontsize=10.5, fontweight="bold", loc="left", pad=12)
    ax.grid(True, color=t["grid"], linewidth=0.8, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=8.5)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.text(0.01, 0.015, "point labels = concurrent requests · Qwen2.5-7B-Instruct on one NVIDIA L4 · 512 in / 256 out",
             color=t["ink2"], fontsize=7.5)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname, facecolor=t["surface"])
    plt.close(fig)


def md_tables(runs):
    levels = sorted({c for v in runs.values() for c in v})
    out = ["### Throughput and latency\n",
           "| Concurrent requests | " + " | ".join(f"{NICE[l]} tok/s" for l in ORDER) +
           " | " + " | ".join(f"{NICE[l]} p95 TTFT" for l in ORDER) + " |",
           "|---" * (1 + 2 * len(ORDER)) + "|"]
    for c in levels:
        row = [str(c)]
        row += [f"{runs[l][c]['output_throughput']:.1f}" for l in ORDER]
        row += [f"{runs[l][c]['p95_ttft_ms']:,.0f} ms" for l in ORDER]
        out.append("| " + " | ".join(row) + " |")

    out += ["", f"### Cost per 1M output tokens (at ${GPU_USD_PER_HOUR:.2f}/GPU-hour)\n",
            "| Concurrent requests | " + " | ".join(NICE[l] for l in ORDER) + " |",
            "|---" * (1 + len(ORDER)) + "|"]
    for c in levels:
        out.append("| " + " | ".join([str(c)] +
                   [f"${cost_per_1m(runs[l][c]['output_throughput']):.2f}" for l in ORDER]) + " |")

    accs = {}
    for f in RAW.glob("acc_*_full.json"):
        d = json.loads(f.read_text())
        accs[d["label"].replace("_full", "")] = d
    if accs:
        out += ["", "### Accuracy (full GSM8K test set)\n",
                "| Setup | Correct | Accuracy | 95% CI |", "|---|---|---|---|"]
        for l in ORDER:
            d = accs.get(l)
            if d:
                lo, hi = d["ci95"]
                out.append(f"| {NICE[l]} | {d['correct']} / {d['n']} | {d['accuracy']:.1%} | {lo:.1%}–{hi:.1%} |")
    return "\n".join(out)


def main():
    runs = load_runs()
    missing = [l for l in ORDER if l not in runs]
    if missing:
        raise SystemExit(f"no results for: {', '.join(missing)}")
    for mode in ("light", "dark"):
        line_chart(runs, mode, "output_throughput", "p95_ttft_ms",
                   "Output throughput (tokens/s)", "p95 time to first token (ms, log scale)",
                   "More throughput costs waiting time, and the curve differs by precision",
                   f"tradeoff-{mode}.png", ylog=True)
        line_chart(runs, mode, "max_concurrency", "median_tpot_ms",
                   "Concurrent requests", "Median time per output token (ms)",
                   "What each user feels as load rises",
                   f"peruser-{mode}.png")
    print(md_tables(runs))
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
