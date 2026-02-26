#!/usr/bin/env python3
"""
Batch-plot PSMC vs TRUE demography for all runs in `psmc_checker/runs`.

For each `*.psmc` under --runs-dir:
- infer `--true-demography` by parent directory name (const/hl/zigzag)
- save `<psmc_stem>.png` next to the `.psmc` (via plot_results.py defaults)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch plot PSMC vs TRUE demography for runs/")

    ap.add_argument(
        "--runs-dir",
        default=str(Path(__file__).resolve().parent / "runs"),
        help="Directory with subfolders containing *.psmc outputs (default: psmc_checker/runs)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate png even if it already exists next to .psmc",
    )
    ap.add_argument(
        "--only-psmc",
        action="store_true",
        help="Plot only the PSMC curve (do not overlay TRUE demography).",
    )
    ap.add_argument(
        "--default-true-demography",
        default=None,
        help="Fallback demography fn name (e.g. get_zigzag) if parent folder is unknown",
    )

    # Forwarded plotting/scaling options (match plot_results.py)
    ap.add_argument("--bin-size", type=int, default=100)
    ap.add_argument(
        "--mu",
        type=float,
        default=None,
        help="Mutation rate per base per generation (default: MU_HUMAN from deepgene/data/data_gen_np.py)",
    )
    ap.add_argument(
        "--generation-time",
        type=float,
        default=25.0,
        help="Time unit multiplier (default: 25.0 => years)",
    )
    ap.add_argument("--psmc-N0", type=float, default=None)
    ap.add_argument("--psmc-time-mult", type=float, default=2.0)
    ap.add_argument("--psmc-size-mult", type=float, default=1.0)

    ap.add_argument("--x-min", type=float, default=None, help="Override X min (default: auto from data)")
    ap.add_argument("--x-max", type=float, default=None, help="Override X max (default: auto from data)")
    ap.add_argument("--y-min", type=float, default=0.0)
    ap.add_argument("--y-max", type=float, default=20_000)
    ap.add_argument("--x-log", action="store_true")

    # --- NN metrics (stubs for future sources) ---
    ap.add_argument(
        "--nn-metrics-source",
        choices=["none", "result_json", "recompute"],
        default="none",
        help="Stub: where to read/compute NN metrics (not implemented yet).",
    )

    return ap.parse_args(argv)


def _infer_true_demography(psmc_path: Path, default_true_demography: Optional[str]) -> Optional[str]:
    folder = psmc_path.parent.name
    mapping = {
        "const": "get_const_demographcs",
        "hl": "get_humanlike",
        "zigzag": "get_zigzag",
    }
    if folder in mapping:
        return mapping[folder]
    if folder.startswith("get_"):
        return folder
    return default_true_demography


def _true_ne_at_psmc_times(*, limits_x: list[float], true_ne: list[float], psmc_times: list[float]) -> np.ndarray:
    """
    Match the plotting convention used in plot_results.py:
    `ax.step(limits_x, true_ne, where="post")` => value true_ne[i] holds for [limits_x[i], limits_x[i+1]).
    """
    if len(limits_x) != len(true_ne):
        raise ValueError(f"Expected len(limits_x)==len(true_ne), got {len(limits_x)} vs {len(true_ne)}")
    if not limits_x:
        return np.array([], dtype=float)

    x = np.asarray(limits_x, dtype=float)
    # msprime demography debug can return (N, 1) trajectory; squeeze to 1D
    y = np.asarray(true_ne, dtype=float).reshape(-1)

    t = np.asarray(psmc_times, dtype=float)
    # idx = last i where x[i] <= t
    idx = np.searchsorted(x, t, side="right") - 1
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx]


def _relative_metrics(ne_true_at: np.ndarray, ne_pred: np.ndarray) -> tuple[float, float, float, int]:
    """
    Compute metrics like in all_in_one.py:
      err = (Ne_true - Ne_pred) / Ne_true
      FMSE = mean(err^2)
      FMAE = mean(|err|)
      RMSE = sqrt(FMSE)
    """
    y_true = np.asarray(ne_true_at, dtype=float)
    y_pred = np.asarray(ne_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: true={y_true.shape}, pred={y_pred.shape}")

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0.0)
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan"), 0

    err = (y_true[mask] - y_pred[mask]) / y_true[mask]
    fmse = float(np.mean(err**2))
    fmae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(fmse)) if math.isfinite(fmse) else float("nan")
    return fmse, rmse, fmae, int(np.sum(mask))


def _write_csv_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    if not runs_dir.exists():
        raise FileNotFoundError(f"--runs-dir does not exist: {runs_dir}")

    # Import here to keep CLI fast and avoid importing matplotlib unless needed
    from psmc_checker import plot_results

    psmc_files = sorted(runs_dir.rglob("*.psmc"))
    if not psmc_files:
        print(f"No .psmc files found under {runs_dir}", file=sys.stderr)
        return 1

    processed = 0
    skipped = 0
    failed = 0

    for psmc_path in psmc_files:
        out_png = psmc_path.with_suffix(".png")
        if out_png.exists() and not args.overwrite:
            skipped += 1
            continue

        true_demo = None
        if not args.only_psmc:
            true_demo = _infer_true_demography(psmc_path, args.default_true_demography)
            if not true_demo:
                print(
                    f"[skip] Cannot infer true demography for {psmc_path} "
                    f"(parent='{psmc_path.parent.name}'). Provide --default-true-demography or use --only-psmc.",
                    file=sys.stderr,
                )
                skipped += 1
                continue

        # Build argv for plot_results.main()
        pr_argv: list[str] = [
            "--psmc",
            str(psmc_path),
            "--bin-size",
            str(args.bin_size),
            "--generation-time",
            str(args.generation_time),
            "--psmc-time-mult",
            str(args.psmc_time_mult),
            "--psmc-size-mult",
            str(args.psmc_size_mult),
            "--y-min",
            str(args.y_min),
            "--y-max",
            str(args.y_max),
        ]
        if args.x_min is not None:
            pr_argv += ["--x-min", str(args.x_min)]
        if args.x_max is not None:
            pr_argv += ["--x-max", str(args.x_max)]
        if true_demo is not None:
            pr_argv += ["--true-demography", true_demo]

        if args.mu is not None:
            pr_argv += ["--mu", str(args.mu)]
        if args.psmc_N0 is not None:
            pr_argv += ["--psmc-N0", str(args.psmc_N0)]
        if args.x_log:
            pr_argv += ["--x-log"]

        try:
            plot_results.main(pr_argv)

            # --- CSV metrics (PSMC vs TRUE) ---
            # Only compute when TRUE is present (i.e. not --only-psmc and we resolved demography).
            if true_demo is not None:
                # Use the same defaults as plot_results.py for TRUE demography params
                dem = plot_results.build_demography(true_demo, initial_size=10_000, random_seed=42)

                limits = list(getattr(plot_results.dgn, "limits"))
                limits_x = [float(t) * float(args.generation_time) for t in limits]
                true_ne = plot_results.demography_to_Ne(dem, limits)

                mu_used = float(args.mu) if args.mu is not None else float(getattr(plot_results.dgn, "MU_HUMAN"))
                psmc_times, psmc_sizes, n0 = plot_results.psmc_parse_times_sizes(
                    psmc_path,
                    bin_size=args.bin_size,
                    mutation_rate=mu_used,
                    generation_time=float(args.generation_time),
                    N0_override=args.psmc_N0,
                    time_mult=float(args.psmc_time_mult),
                    size_mult=float(args.psmc_size_mult),
                )

                ne_true_at = _true_ne_at_psmc_times(
                    limits_x=limits_x,
                    true_ne=true_ne,
                    psmc_times=psmc_times,
                )

                fmse, rmse, fmae, n_points = _relative_metrics(ne_true_at, np.asarray(psmc_sizes, dtype=float))

                out_csv = psmc_path.with_suffix(".csv")
                row = {
                    "psmc_path": str(psmc_path),
                    "png_path": str(out_png),
                    "true_demography": true_demo,
                    "bin_size": int(args.bin_size),
                    "mu": mu_used,
                    "generation_time": float(args.generation_time),
                    "N0": float(n0),
                    "n_points": int(n_points),
                    "FMSE": fmse,
                    "RMSE": rmse,
                    "FMAE": fmae,
                    # --- stubs for future NN metrics (sources 2 & 3) ---
                    "nn_metrics_source": ("" if args.nn_metrics_source == "none" else args.nn_metrics_source),
                    "nn_FMSE": "",
                    "nn_RMSE": "",
                    "nn_FMAE": "",
                }
                if args.nn_metrics_source != "none":
                    print(
                        f"[warn] NN metrics source '{args.nn_metrics_source}' is not implemented yet; "
                        f"leaving nn_* fields empty for {psmc_path}",
                        file=sys.stderr,
                    )
                _write_csv_row(out_csv, row)

            processed += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[fail] {psmc_path}: {type(e).__name__}: {e}", file=sys.stderr)

    print(
        f"Done. processed={processed} skipped={skipped} failed={failed} runs_dir={runs_dir}",
        file=sys.stderr if failed else sys.stdout,
    )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

