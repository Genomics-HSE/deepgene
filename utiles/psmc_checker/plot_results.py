#!/usr/bin/env python3
"""
Plot PSMC vs TRUE demography vs NN reconstruction (Ne).

This script is a merged/cleaned version of the old plotting logic that was split across:
- `__old_code/.../plot_results.py` (PSMC parsing)
- `__old_code/.../NN_test/analyze.py` (priors->Ne conversion + True demography plotting)

It intentionally avoids importing `deepgene.data` package (which pulls torch) and instead
loads `deepgene/data/data_gen_np.py` directly.

Inputs (choose what you have):
- PSMC output `.psmc` (required for PSMC curve)
- TRUE demography name (optional): e.g. get_const_demographcs / get_humanlike / get_zigzag
- NN curve (optional):
  - either `--nn-priors-json` (prob mass per interval) OR
  - `--nn-ne-json` (already computed Ne per interval)
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_deepgene_data_gen_np():
    """Load `deepgene/data/data_gen_np.py` without importing `deepgene.data`."""
    cur = Path(__file__).resolve()
    module_path = None
    for parent in [cur.parent, *cur.parents]:
        candidate = parent / "deepgene" / "data" / "data_gen_np.py"
        if candidate.exists():
            module_path = candidate
            break
    if module_path is None:
        raise ImportError(
            "Cannot locate deepgene/data/data_gen_np.py by walking upwards from "
            f"{Path(__file__).resolve()}"
        )
    spec = importlib.util.spec_from_file_location(
        "deepgene_data_gen_np_standalone", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


dgn = _load_deepgene_data_gen_np()


def _resolve_demography_fn(name: str) -> Callable:
    fn = getattr(dgn, name, None)
    if fn is None or not callable(fn):
        raise ValueError(
            f"Unknown demography '{name}'. Expected a callable in deepgene.data.data_gen_np."
        )
    return fn


def build_demography(
    demography_name: str,
    *,
    initial_size: int,
    random_seed: int,
):
    fn = _resolve_demography_fn(demography_name)
    sig = inspect.signature(fn)
    kwargs = {}
    if "initial_size" in sig.parameters:
        kwargs["initial_size"] = initial_size
    if "population" in sig.parameters:
        kwargs["population"] = initial_size
    if "random_seed" in sig.parameters:
        kwargs["random_seed"] = random_seed
    return fn(**kwargs)


def demography_to_Ne(demography, limits: Sequence[float]) -> list[float]:
    debug = demography.debug()
    # msprime returns a numpy array
    return list(debug.population_size_trajectory(limits))


def psmc_parse_times_sizes(
    psmc_path: str | Path,
    *,
    bin_size: int,
    mutation_rate: float,
    generation_time: float,
    N0_override: Optional[float] = None,
    time_mult: float = 2.0,
    size_mult: float = 1.0,
) -> Tuple[list[float], list[float], float]:
    """
    Parse .psmc output and return scaled times/sizes.

    Default scaling matches classic scripts:
      times = generation_time * (time_mult * N0) * time_window
      sizes = (size_mult * N0) * lambda

    Old variants:
    - `psmc2fun`: time_mult=2.0, size_mult=1.0
    - `psmc2fun2` (\"msprime 100 in 1\" hack): time_mult=1.0, size_mult=0.5
    """
    p = Path(psmc_path)
    text = p.read_text(encoding="utf-8", errors="ignore")

    # time windows + lambdas from the last iteration
    last_block = text.split("//\n")[-2].split("\n")
    time_windows: list[float] = []
    lambdas: list[float] = []
    for line in last_block:
        if line.startswith("RS"):
            parts = line.split("\t")
            # RS  <k>  <t_k>  <lambda_k> ...
            time_windows.append(float(parts[2]))
            lambdas.append(float(parts[3]))

    # theta from the last PA line
    pa_tail = text.split("PA\t")[-1].split("\n", 1)[0].split()
    theta = float(pa_tail[1])

    if N0_override is None:
        N0 = theta / (4 * mutation_rate) / bin_size
    else:
        N0 = float(N0_override)

    times = [generation_time * (time_mult * N0) * t for t in time_windows]
    sizes = [(size_mult * N0) * lam for lam in lambdas]

    return times, sizes, N0


def from_probability_to_N(priors: Sequence[float], limits: Sequence[float]) -> list[float]:
    """
    Convert probability mass per interval to Ne (PSMC-like) using the same formula as old `analyze.py`.
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
    ap = argparse.ArgumentParser(description="Plot PSMC vs TRUE vs NN Ne curves.")

    ap.add_argument("--psmc", required=True, help="Path to .psmc output file")
    ap.add_argument("--out", default=None, help="Output image path (default: next to --psmc)")

    # PSMC scaling
    ap.add_argument("--bin-size", type=int, default=100, help="PSMC bin size (usually 100)")
    ap.add_argument(
        "--mu",
        type=float,
        default=2.5e-8,
        help="Mutation rate per base per generation (default: 2.5e-8)",
    )
    ap.add_argument(
        "--generation-time",
        type=float,
        default=25.0,
        help="Time unit multiplier (default: 25.0 => years if demography times are in generations)",
    )
    ap.add_argument("--psmc-N0", type=float, default=None, help="Override N0 (otherwise computed from theta)")
    ap.add_argument("--psmc-time-mult", type=float, default=2.0, help="Time scaling multiplier (default 2.0)")
    ap.add_argument("--psmc-size-mult", type=float, default=1.0, help="Size scaling multiplier (default 1.0)")

    # TRUE curve (deepgene demography)
    ap.add_argument("--true-demography", default=None, help="Demography fn name from deepgene.data.data_gen_np (e.g. get_zigzag)")
    ap.add_argument("--true-initial-size", type=int, default=10_000, help="Initial size passed into demography")
    ap.add_argument("--true-random-seed", type=int, default=42, help="Random seed passed into demography (if supported)")

    # NN curve
    ap.add_argument("--nn-priors-json", default=None, help="JSON file containing priors/probabilities per interval")
    ap.add_argument("--nn-ne-json", default=None, help="JSON file containing Ne values per interval")
    ap.add_argument("--nn-key", default=None, help="Optional key inside the JSON (e.g. 'priors' or 'Ne')")

    # Plot options
    ap.add_argument("--x-min", type=float, default=None, help="Override X min (default: auto from data)")
    ap.add_argument("--x-max", type=float, default=None, help="Override X max (default: auto from data)")
    ap.add_argument("--y-min", type=float, default=0.0)
    ap.add_argument("--y-max", type=float, default=20_000)
    ap.add_argument("--x-log", action="store_true", help="Use log scale for X")
    ap.add_argument("--show", action="store_true", help="Show interactive window")

    return ap.parse_args(argv)


def _load_json_array(path: str | Path, key: Optional[str]) -> list[float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if key is None:
        if isinstance(data, list):
            return [float(x) for x in data]
        raise ValueError("JSON root is not a list; provide --nn-key")
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {path}")
    arr = data[key]
    if not isinstance(arr, list):
        raise ValueError(f"JSON key '{key}' is not a list in {path}")
    return [float(x) for x in arr]


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    limits = list(getattr(dgn, "limits"))
    # Keep X units consistent across curves:
    # - PSMC times are already scaled by generation_time
    # - TRUE/NN curves are defined on `limits` (deepgene time grid), so scale them too
    limits_x = [float(t) * float(args.generation_time) for t in limits]

    psmc_path = Path(args.psmc)
    out_path = Path(args.out) if args.out else psmc_path.with_suffix(".png")

    # --- Curves ---
    psmc_times, psmc_sizes, N0 = psmc_parse_times_sizes(
        psmc_path,
        bin_size=args.bin_size,
        mutation_rate=args.mu,
        generation_time=args.generation_time,
        N0_override=args.psmc_N0,
        time_mult=args.psmc_time_mult,
        size_mult=args.psmc_size_mult,
    )

    true_Ne: Optional[list[float]] = None
    if args.true_demography:
        dem = build_demography(
            args.true_demography,
            initial_size=args.true_initial_size,
            random_seed=args.true_random_seed,
        )
        true_Ne = demography_to_Ne(dem, limits)

    nn_Ne: Optional[list[float]] = None
    if args.nn_ne_json:
        nn_Ne = _load_json_array(args.nn_ne_json, args.nn_key)
    elif args.nn_priors_json:
        priors = _load_json_array(args.nn_priors_json, args.nn_key)
        nn_Ne = from_probability_to_N(priors, limits)

    # --- Plot ---
    fig, ax = plt.subplots()

    # Get recombination rate for label
    rho = float(getattr(dgn, "RHO_HUMAN", 1.6e-8))
    # Format mutation and recombination rates for label (scientific notation)
    def format_rate(r):
        """Format rate in scientific notation, e.g., 2.5e-8 -> 2.50e-8"""
        s = f"{r:.2e}"
        # Remove leading zeros from exponent
        if "e-" in s:
            base, exp = s.split("e-")
            exp_clean = exp.lstrip("0") or "0"
            return f"{base}e-{exp_clean}"
        elif "e+" in s:
            base, exp = s.split("e+")
            exp_clean = exp.lstrip("0") or "0"
            return f"{base}e+{exp_clean}"
        else:
            return s
    
    mu_str = format_rate(args.mu)
    rho_str = format_rate(rho)
    psmc_label = f"PSMC (N0={N0:.2f}, mu={mu_str}, rho={rho_str})"
    
    ax.step(psmc_times, psmc_sizes, where="post", linestyle="--", color="tab:blue", label=psmc_label)

    if true_Ne is not None:
        ax.step(limits_x, true_Ne, where="post", color="tab:red", label="TRUE (demography)")

    if nn_Ne is not None:
        ax.step(
            limits_x,
            nn_Ne + [nn_Ne[-1]],
            where="post",
            color="black",
            linestyle="-.",
            linewidth=1.5,
            label="NN",
        )

    x_label = "Time (years)" if float(args.generation_time) != 1.0 else "Time (generations)"
    ax.set_xlabel(x_label)
    ax.set_ylabel("N_e")
    ax.grid(True)

    # --- X axis adaptation for readability ---
    # If the time span is very wide, use log scale automatically unless user forced linear.
    x_min = args.x_min
    x_max = args.x_max
    xs_pos = [float(x) for x in psmc_times if float(x) > 0.0]
    data_min = min(xs_pos) if xs_pos else None
    data_max = max(xs_pos) if xs_pos else None

    use_log = bool(args.x_log)
    if not use_log and data_min is not None and data_max is not None and x_min is None and x_max is None:
        # Heuristic: if the visible range spans >= 50x, log is much more readable.
        if data_min > 0.0 and (data_max / data_min) >= 50.0:
            use_log = True

    if use_log:
        ax.set_xscale("log")

    # Auto x-limits (unless overridden)
    if x_min is None or x_max is None:
        if data_min is not None and data_max is not None:
            if x_min is None:
                x_min = max(data_min / 1.5, 1e-9) if use_log else 0.0
            if x_max is None:
                x_max = data_max * 1.05
        else:
            x_min = 1e-3 if use_log else 0.0
            x_max = 1.0

    ax.set_xlim(float(x_min), float(x_max))
    ax.set_ylim(args.y_min, args.y_max)
    ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)

    if args.show:
        plt.show()
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

