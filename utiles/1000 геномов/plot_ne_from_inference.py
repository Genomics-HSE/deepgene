#!/usr/bin/env python3
"""
Plot Ne curve from NN inference results (nn_priors.json).

This script converts priors from nn_priors.json to Ne values and plots them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_deepgene_data_gen_np():
    """Load `deepgene/data/data_gen_np.py` without importing `deepgene.data`."""
    cur = Path(__file__).resolve()
    repo_root = cur.parents[1]
    module_path = repo_root / "deepgene" / "data" / "data_gen_np.py"
    if not module_path.exists():
        raise ImportError(f"Cannot locate deepgene/data/data_gen_np.py at {module_path}")
    spec = importlib.util.spec_from_file_location("deepgene_data_gen_np_standalone", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def from_probability_to_N(priors: Sequence[float], limits: Sequence[float]) -> list[float]:
    """
    Convert probability mass per interval to Ne (PSMC-like).
    """
    priors = list(priors)
    tau = np.array(limits[1:], dtype=float) - np.array(limits[:-1], dtype=float)
    if len(priors) != len(tau):
        raise ValueError(
            f"Expected priors length {len(tau)} (len(limits)-1), got {len(priors)}"
        )

    l = 1.0
    N: list[float] = []
    for j in range(len(tau)):
        pj = float(priors[j])
        # Guard: (1 - pj/l) must be in (0, 1)
        eps = 1e-12
        val = max(min(1.0 - pj / l, 1.0 - eps), eps)
        Nj = float(np.round(tau[j] / (-2.0 * np.log(val))))
        N.append(Nj)
        l *= float(np.exp(-tau[j] / (2.0 * Nj)))
    return N


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot Ne curve from NN inference priors.")
    ap.add_argument("--nn-priors-json", required=True, help="Path to nn_priors.json from inference")
    ap.add_argument("--out", default=None, help="Output image path (default: next to JSON with .png)")
    ap.add_argument(
        "--generation-time",
        type=float,
        default=25.0,
        help="Time unit multiplier (default: 25.0 => years if demography times are in generations)",
    )
    ap.add_argument("--nn-key", default="priors", help="Key in JSON containing priors array (default: 'priors')")
    ap.add_argument("--x-min", type=float, default=None, help="X axis minimum (default: auto)")
    ap.add_argument("--x-max", type=float, default=None, help="X axis maximum (default: auto)")
    ap.add_argument("--y-min", type=float, default=0.0, help="Y axis minimum (default: 0.0)")
    ap.add_argument("--y-max", type=float, default=None, help="Y axis maximum (default: auto)")
    ap.add_argument("--x-log", action="store_true", help="Use log scale for X axis")
    ap.add_argument("--show", action="store_true", help="Show interactive window")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    
    # Load deepgene limits
    dgn = _load_deepgene_data_gen_np()
    limits = list(getattr(dgn, "limits"))
    limits_x = [float(t) * float(args.generation_time) for t in limits]
    
    # Load priors from JSON
    priors_path = Path(args.nn_priors_json)
    data = json.loads(priors_path.read_text(encoding="utf-8"))
    
    if args.nn_key not in data:
        raise KeyError(f"Key '{args.nn_key}' not found in {priors_path}")
    priors = data[args.nn_key]
    if not isinstance(priors, list):
        raise ValueError(f"Key '{args.nn_key}' is not a list in {priors_path}")
    priors = [float(x) for x in priors]
    
    # Convert priors to Ne
    nn_Ne = from_probability_to_N(priors, limits)
    
    # Determine output path
    out_path = Path(args.out) if args.out else priors_path.parent / "ne_plot.png"
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.step(
        limits_x,
        nn_Ne + [nn_Ne[-1]],
        where="post",
        color="black",
        linestyle="-",
        linewidth=2.0,
        label="Ne (NN inference)",
    )
    
    x_label = "Time (years)" if float(args.generation_time) != 1.0 else "Time (generations)"
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("N_e (Effective Population Size)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)
    
    # Set X axis scale
    if args.x_log:
        ax.set_xscale("log")
    
    # Set axis limits
    if args.x_min is not None or args.x_max is not None:
        x_min = args.x_min if args.x_min is not None else min(limits_x)
        x_max = args.x_max if args.x_max is not None else max(limits_x)
        ax.set_xlim(x_min, x_max)
    
    y_max = args.y_max if args.y_max is not None else max(nn_Ne) * 1.1 if nn_Ne else 20000
    ax.set_ylim(args.y_min, y_max)
    
    # Title with sample info
    title_parts = []
    if "checkpoint" in data:
        checkpoint_name = Path(data["checkpoint"]).stem
        title_parts.append(f"Model: {checkpoint_name}")
    if "n_windows" in data:
        title_parts.append(f"Windows: {data['n_windows']}")
    if title_parts:
        ax.set_title(" | ".join(title_parts), fontsize=10)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"✓ График сохранен: {out_path}")
    
    if args.show:
        plt.show()
    plt.close(fig)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
