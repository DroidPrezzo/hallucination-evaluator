#!/usr/bin/env python3
"""Analyze a completed run directory.

Produces, from generations + judgments/verifications:
  - analysis.md      : rate-vs-length tables with bootstrap 95% CIs, EFC per
                       model, fresh-vs-legacy contamination gap, cost per model,
                       and a cross-judge Cohen's kappa table when >=2 judges ran.
  - curves.png       : hallucination rate vs context length, one line per model,
                       with CI bands.
  - raw_summary.csv  : per (model, contamination, length) rate + CI.

Usage:
    python scripts/analyze.py results/runs/<run_id> [--baseline-length 1000]
        [--delta 0.10] [--bootstrap 1000] [--seed 0] [--judge-spec <spec>]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.persistence import RunStore  # noqa: E402

# Validated categorical palette (dataviz skill reference instance, light mode),
# assigned to models in fixed slot order — never cycled cosmetically. Worst
# adjacent CVD deltaE 24.2 (>= 12 target).
_PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
            "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
_INK = "#0b0b0b"
_INK2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_SURFACE = "#fcfcfb"
_PLANE = "#f9f9f7"

# Contamination class -> line style (secondary, non-color encoding).
_CONTAM_STYLE = {"fresh2026": "-", "public_legacy": "--"}


def _short_spec(spec: str) -> str:
    """Readable label: drop query params, keep provider:model."""
    return spec.split("?", 1)[0]


def _fmt_tokens(n: float) -> str:
    n = int(n)
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_observations(store: RunStore, config: dict, judge_spec: str):
    """One observation per judged summary: model, length, contamination, rate."""
    gens = {g["idempotency_key"]: g for g in store.load_generations()}
    mode = config.get("judge_mode", "holistic")
    obs: list[dict] = []

    if mode == "holistic":
        for j in store.load_judgments():
            if j.get("judge_spec") != judge_spec or j.get("verdict") is None:
                continue
            g = gens.get(j["generation_key"])
            if g is None:
                continue
            obs.append({
                "model_spec": g["model_spec"],
                "length": g["requested_context_tokens"],
                "contamination_risk": g.get("contamination_risk", "unknown"),
                "doc_id": g["doc_id"],
                "rate": 1.0 if j["verdict"] else 0.0,
            })
    else:
        counts: dict[str, Counter] = defaultdict(Counter)
        for v in store.load_verifications():
            if v.get("judge_spec") != judge_spec:
                continue
            if v.get("verdict") in ("supported", "unsupported", "ambiguous"):
                counts[v["generation_key"]][v["verdict"]] += 1
        for gen_key, c in counts.items():
            total = sum(c.values())
            g = gens.get(gen_key)
            if total == 0 or g is None:
                continue
            obs.append({
                "model_spec": g["model_spec"],
                "length": g["requested_context_tokens"],
                "contamination_risk": g.get("contamination_risk", "unknown"),
                "doc_id": g["doc_id"],
                "rate": c["unsupported"] / total,
            })
    return obs, mode


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _cell_seed(base_seed: int, *parts) -> int:
    """Deterministic per-cell seed so a cell's CI is identical wherever it is
    computed (analysis.md and raw_summary.csv agree), independent of order."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return (base_seed + int(h[:8], 16)) % (2 ** 32)


def bootstrap_ci(rates: list[float], n_resamples: int, seed: int,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    arr = np.asarray(rates, dtype=float)
    mean = float(arr.mean()) if arr.size else float("nan")
    if arr.size <= 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    return (mean,
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def compute_efc(length_to_rate: dict[int, float], baseline_length: int,
                delta: float) -> dict:
    """Effective Faithful Context: largest tested length L with
    rate(L) <= rate(baseline) + delta."""
    lengths = sorted(length_to_rate)
    if baseline_length in length_to_rate:
        base = baseline_length
    else:
        base = min(lengths, key=lambda L: abs(L - baseline_length))
    baseline_rate = length_to_rate[base]
    threshold = baseline_rate + delta
    qualifying = [L for L in lengths if length_to_rate[L] <= threshold + 1e-12]
    return {
        "efc": max(qualifying) if qualifying else base,
        "baseline_length_used": base,
        "baseline_rate": baseline_rate,
        "threshold": threshold,
    }


def cohens_kappa(pairs: list[tuple]) -> tuple[float, int]:
    n = len(pairs)
    if n == 0:
        return float("nan"), 0
    cats = sorted({lbl for p in pairs for lbl in p})
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if abs(1 - pe) < 1e-12:
        return (1.0 if po >= 1.0 - 1e-12 else 0.0), n
    return (po - pe) / (1 - pe), n


def kappa_table(store: RunStore, config: dict) -> Optional[list[dict]]:
    judges = config.get("judge_specs") or []
    if len(judges) < 2:
        return None
    mode = config.get("judge_mode", "holistic")
    verdicts: dict = defaultdict(dict)
    if mode == "holistic":
        for j in store.load_judgments():
            if j.get("verdict") is not None:
                verdicts[j["generation_key"]][j["judge_spec"]] = bool(j["verdict"])
    else:
        for v in store.load_verifications():
            if v.get("verdict") in ("supported", "unsupported", "ambiguous"):
                verdicts[(v["generation_key"], v["claim_id"])][v["judge_spec"]] = v["verdict"]

    rows = []
    for i in range(len(judges)):
        for k in range(i + 1, len(judges)):
            ja, jb = judges[i], judges[k]
            pairs = [(d[ja], d[jb]) for d in verdicts.values() if ja in d and jb in d]
            kappa, n = cohens_kappa(pairs)
            rows.append({"judge_a": ja, "judge_b": jb, "kappa": kappa, "n_items": n})
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_cells(obs: list[dict], n_boot: int, base_seed: int,
                    by_contamination: bool):
    """Return {model: {length: {mean, lo, hi, n}}}, aggregating over documents.

    by_contamination=False pools contamination classes into one cell per
    (model, length); True keys cells by (model, contamination)."""
    groups: dict = defaultdict(lambda: defaultdict(list))
    for o in obs:
        model_key = (o["model_spec"], o["contamination_risk"]) if by_contamination else o["model_spec"]
        groups[model_key][o["length"]].append(o["rate"])

    out: dict = {}
    for model_key, lengths in groups.items():
        out[model_key] = {}
        for length, rates in sorted(lengths.items()):
            if by_contamination:
                model, contam = model_key
            else:
                model, contam = model_key, "all"
            seed = _cell_seed(base_seed, model, contam, length)
            mean, lo, hi = bootstrap_ci(rates, n_boot, seed)
            out[model_key][length] = {"mean": mean, "lo": lo, "hi": hi, "n": len(rates)}
    return out


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def plot_curves(cells: dict, out_path: Path, title: str, multi_contam: bool) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(_PLANE)
    ax.set_facecolor(_SURFACE)

    # Stable model->color assignment (color follows the model entity).
    models = sorted({k[0] if isinstance(k, tuple) else k for k in cells})
    color_of = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(models)}
    if len(models) > len(_PALETTE):
        print(f"  warning: {len(models)} models exceed the 8-hue palette; colors reused",
              file=sys.stderr)

    all_lengths: set = set()
    max_hi = 0.0
    for key, series in sorted(cells.items()):
        model = key[0] if isinstance(key, tuple) else key
        contam = key[1] if isinstance(key, tuple) else None
        xs = sorted(series)
        all_lengths.update(xs)
        means = [series[x]["mean"] for x in xs]
        los = [series[x]["lo"] for x in xs]
        his = [series[x]["hi"] for x in xs]
        max_hi = max([max_hi] + his)
        color = color_of[model]
        style = _CONTAM_STYLE.get(contam, "-") if multi_contam else "-"
        label = f"{_short_spec(model)} · {contam}" if multi_contam else _short_spec(model)
        ax.fill_between(xs, los, his, color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, means, color=color, linewidth=2, linestyle=style, marker="o",
                markersize=7, markeredgecolor=_SURFACE, markeredgewidth=1.2,
                label=label, zorder=3)

    ax.set_xscale("log")
    ticks = sorted(all_lengths)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_tokens(v)))
    ax.minorticks_off()
    ax.set_ylim(0, min(1.0, max(0.05, max_hi * 1.15)))

    ax.set_xlabel("Requested context length (tokens)", color=_INK2, fontsize=11)
    ax.set_ylabel("Hallucination rate", color=_INK2, fontsize=11)
    ax.set_title(title, color=_INK, fontsize=12, pad=12)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_AXIS)
    ax.tick_params(colors=_MUTED, labelsize=10)

    if len(cells) >= 2:
        ax.legend(frameon=False, fontsize=9, labelcolor=_INK2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_raw_csv(obs: list[dict], out_path: Path, n_boot: int,
                  base_seed: int) -> None:
    # Per (model, contamination, length) plus a contamination='all' aggregate.
    # Per-cell seeds match aggregate_cells so CIs agree with analysis.md.
    rows: list[list] = []
    by_mcl: dict = defaultdict(list)
    by_ml: dict = defaultdict(list)
    for o in obs:
        by_mcl[(o["model_spec"], o["contamination_risk"], o["length"])].append(o["rate"])
        by_ml[(o["model_spec"], o["length"])].append(o["rate"])

    for (model, contam, length), rates in sorted(by_mcl.items()):
        mean, lo, hi = bootstrap_ci(rates, n_boot, _cell_seed(base_seed, model, contam, length))
        rows.append([model, contam, length, len(rates), f"{mean:.6f}", f"{lo:.6f}", f"{hi:.6f}"])
    for (model, length), rates in sorted(by_ml.items()):
        mean, lo, hi = bootstrap_ci(rates, n_boot, _cell_seed(base_seed, model, "all", length))
        rows.append([model, "all", length, len(rates), f"{mean:.6f}", f"{lo:.6f}", f"{hi:.6f}"])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model_spec", "contamination_risk", "context_length",
                    "n_docs", "mean_rate", "ci_low", "ci_high"])
        w.writerows(rows)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_analysis_md(path: Path, config: dict, mode: str, judge_spec: str,
                      pooled: dict, efc_by_model: dict, obs: list[dict],
                      cost_totals: dict, kappa_rows: Optional[list[dict]],
                      baseline_length: int, delta: float, n_boot: int,
                      seed: int) -> None:
    lines: list[str] = []
    lines.append(f"# Analysis — run `{config.get('run_id', '?')}`\n")
    lines.append("## Parameters (on the record)\n")
    lines.append(_md_table(
        ["key", "value"],
        [
            ["judge_mode", mode],
            ["primary_judge", judge_spec],
            ["judges", ", ".join(config.get("judge_specs") or [])],
            ["decomposer_spec", str(config.get("decomposer_spec"))],
            ["document_source", (config.get("document_source") or {}).get("name", "?")],
            ["git_commit", config.get("git_commit", "?")],
            ["EFC baseline_length", baseline_length],
            ["EFC delta", delta],
            ["bootstrap_resamples", n_boot],
            ["bootstrap_seed", seed],
        ],
    ))
    lines.append("")

    lengths = sorted({length for m in pooled.values() for length in m})
    lines.append("## Hallucination rate vs context length\n")
    lines.append(f"Mean per-summary hallucination rate with bootstrap 95% CI "
                 f"(resampled over documents, n={n_boot}). Cell: `mean [lo, hi] (n)`.\n")
    header = ["model"] + [_fmt_tokens(L) for L in lengths]
    rows = []
    for model in sorted(pooled):
        cells = []
        for L in lengths:
            c = pooled[model].get(L)
            cells.append(f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}] (n={c['n']})"
                         if c else "—")
        rows.append([_short_spec(model)] + cells)
    lines.append(_md_table(header, rows))
    lines.append("")

    lines.append("## Effective Faithful Context (EFC)\n")
    lines.append(f"Largest tested length L where rate(L) ≤ rate({baseline_length}) + "
                 f"{delta:.2f}.\n")
    rows = []
    for model in sorted(efc_by_model):
        e = efc_by_model[model]
        rows.append([
            _short_spec(model), _fmt_tokens(e["efc"]),
            _fmt_tokens(e["baseline_length_used"]),
            f"{e['baseline_rate']:.3f}", f"{e['threshold']:.3f}",
        ])
    lines.append(_md_table(
        ["model", "EFC", "baseline_len_used", "baseline_rate", "threshold"], rows))
    lines.append("")

    # Contamination gap
    lines.append("## Contamination gap (fresh vs legacy)\n")
    contam_classes = sorted({o["contamination_risk"] for o in obs})
    if len(contam_classes) < 2:
        lines.append(f"Only one contamination class present "
                     f"(`{contam_classes[0] if contam_classes else 'none'}`); "
                     f"gap not computable.\n")
    else:
        by_mc: dict = defaultdict(list)
        for o in obs:
            by_mc[(o["model_spec"], o["contamination_risk"])].append(o["rate"])
        models = sorted({o["model_spec"] for o in obs})
        header = ["model"] + contam_classes
        gap_col = "fresh2026" in contam_classes and "public_legacy" in contam_classes
        if gap_col:
            header.append("gap (fresh−legacy)")
        rows = []
        for model in models:
            cells = []
            means = {}
            for cc in contam_classes:
                rates = by_mc.get((model, cc))
                means[cc] = sum(rates) / len(rates) if rates else None
                cells.append(f"{means[cc]:.3f}" if means[cc] is not None else "—")
            if gap_col:
                f_, l_ = means.get("fresh2026"), means.get("public_legacy")
                cells.append(f"{f_ - l_:+.3f}" if (f_ is not None and l_ is not None) else "—")
            rows.append([_short_spec(model)] + cells)
        lines.append(_md_table(header, rows))
    lines.append("")

    lines.append("## Cost per model (USD)\n")
    if cost_totals:
        rows = [[label, f"{cost:.4f}"] for label, cost in sorted(cost_totals.items())]
        rows.append(["**total**", f"{sum(cost_totals.values()):.4f}"])
        lines.append(_md_table(["component", "cost_usd"], rows))
    else:
        lines.append("No cost estimates recorded (local models or missing pricing).")
    lines.append("")

    lines.append("## Cross-judge agreement (Cohen's kappa)\n")
    if kappa_rows is None:
        lines.append("Fewer than two judges ran; kappa not computed.")
    else:
        level = "claim" if mode == "claims" else "summary"
        lines.append(f"Agreement at the {level} level.\n")
        rows = [[_short_spec(r["judge_a"]), _short_spec(r["judge_b"]),
                 f"{r['kappa']:.3f}" if r["kappa"] == r["kappa"] else "n/a",
                 r["n_items"]] for r in kappa_rows]
        lines.append(_md_table(["judge A", "judge B", "kappa", "n_items"], rows))
    lines.append("")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------

def analyze_run(run_dir, *, baseline_length: int = 1000, delta: float = 0.10,
                bootstrap: int = 1000, seed: int = 0,
                judge_spec: Optional[str] = None, output_dir=None):
    """Analyze a run directory; writes analysis.md, curves.png, raw_summary.csv.

    Returns (out_dir, info). Raises ValueError if there is nothing to analyze."""
    n_boot = max(1000, bootstrap)
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    run_id = config.get("run_id", run_dir.name)
    store = RunStore(run_id, str(run_dir.parent.parent))

    judge_spec = judge_spec or (config.get("judge_specs") or [None])[0]
    if judge_spec is None:
        raise ValueError("No judge spec in config; pass judge_spec.")

    obs, mode = collect_observations(store, config, judge_spec)
    if not obs:
        raise ValueError(f"No judged summaries for judge '{judge_spec}' in {run_dir}.")

    out_dir = Path(output_dir) if output_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled = aggregate_cells(obs, n_boot, seed, by_contamination=False)
    efc_by_model = {
        model: compute_efc({L: c["mean"] for L, c in series.items()},
                           baseline_length, delta)
        for model, series in pooled.items()
    }

    contam_classes = {o["contamination_risk"] for o in obs}
    multi_contam = len(contam_classes) > 1
    plot_cells = (aggregate_cells(obs, n_boot, seed, by_contamination=True)
                  if multi_contam else pooled)

    cost_totals = store.compute_cost_totals()
    kappa_rows = kappa_table(store, config)

    title = f"Faithfulness vs context — {mode} judge ({_short_spec(judge_spec)})"
    plot_curves(plot_cells, out_dir / "curves.png", title, multi_contam)
    write_raw_csv(obs, out_dir / "raw_summary.csv", n_boot, seed)
    write_analysis_md(out_dir / "analysis.md", config, mode, judge_spec, pooled,
                      efc_by_model, obs, cost_totals, kappa_rows,
                      baseline_length, delta, n_boot, seed)

    return out_dir, {"mode": mode, "judge": judge_spec, "models": len(pooled),
                     "observations": len(obs), "bootstrap": n_boot}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Path to results/runs/<run_id>")
    parser.add_argument("--baseline-length", type=int, default=1000)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Bootstrap resamples (>=1000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--judge-spec", default=None,
                        help="Primary judge for rate tables (default: first in config)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write outputs (default: the run dir)")
    args = parser.parse_args()

    try:
        out_dir, info = analyze_run(
            args.run_dir, baseline_length=args.baseline_length, delta=args.delta,
            bootstrap=args.bootstrap, seed=args.seed, judge_spec=args.judge_spec,
            output_dir=args.output_dir)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))

    print(f"Wrote analysis.md, curves.png, raw_summary.csv to {out_dir}")
    print(f"  judge={info['judge']} mode={info['mode']} models={info['models']} "
          f"observations={info['observations']} bootstrap={info['bootstrap']}")


if __name__ == "__main__":
    main()
