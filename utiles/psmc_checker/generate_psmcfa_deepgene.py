#!/usr/bin/env python3
"""
Generate PSMC input files (.psmcfa) using the *current* deepgene demography helpers.

This script intentionally writes ONLY the PSMC format (no NN text, no priors).

Key idea:
- We simulate a binned sequence of length ceil(sequence_length / bin_size).
- Each bin is marked as:
  - 'K' if at least one mutation falls into that bin
  - 'T' otherwise

The output is FASTA-like (as expected by PSMC): records with a header line starting with '>'
and a sequence split into fixed-width lines.
"""

from __future__ import annotations

import argparse
import inspect
import importlib.util
import math
import os
from pathlib import Path
from typing import Callable, Optional

def _load_deepgene_data_gen_np():
    """
    Load `deepgene/data/data_gen_np.py` without importing `deepgene.data` package.

    Why: `deepgene/data/__init__.py` imports torch via DatasetXY, but torch is not needed
    for PSMCFA generation and may be unavailable in lightweight environments.
    """
    # Find repository root by walking upwards until we see `deepgene/data/data_gen_np.py`.
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


def _wrap_fasta(seq: str, line_length: int) -> str:
    if line_length <= 0:
        return seq + "\n"
    out = []
    for i in range(0, len(seq), line_length):
        out.append(seq[i : i + line_length])
    return "\n".join(out) + "\n"


def _resolve_demography_fn(name: str) -> Callable:
    fn = getattr(dgn, name, None)
    if fn is None or not callable(fn):
        raise ValueError(
            f"Unknown demography '{name}'. Expected a callable in deepgene.data.data_gen_np."
        )
    return fn


def _build_demography(
    demography_name: str,
    *,
    initial_size: int,
    random_seed: int,
) -> "dgn.msprime.Demography":
    fn = _resolve_demography_fn(demography_name)
    sig = inspect.signature(fn)

    kwargs = {}
    if "initial_size" in sig.parameters:
        kwargs["initial_size"] = initial_size
    if "population" in sig.parameters:
        kwargs["population"] = initial_size
    if "random_seed" in sig.parameters:
        kwargs["random_seed"] = random_seed

    return fn(**kwargs)  # type: ignore[no-any-return]


def _write_record(handle, header: str, seq: str, *, line_length: int) -> None:
    handle.write(f">{header}\n")
    handle.write(_wrap_fasta(seq, line_length=line_length))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate .psmcfa using demographies from the current deepgene package."
    )

    parser.add_argument(
        "--demography",
        type=str,
        default="get_const_demographcs",
        help="Demography function name from deepgene.data.data_gen_np (e.g. get_const_demographcs, get_humanlike, get_zigzag, get_set1...).",
    )
    parser.add_argument(
        "--initial-size",
        type=int,
        default=10_000,
        help="Base population size parameter passed to demography (as initial_size or population, depending on function signature).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed forwarded to demography generator (if it accepts random_seed).",
    )

    parser.add_argument(
        "--num-replicates",
        type=int,
        default=100,
        help="How many independent replicates to generate.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=30_000_000,
        help="Ungrouped sequence length (e.g. 3_000_000 or 30_000_000). The produced .psmcfa length will be ceil(sequence_length / bin_size).",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=100,
        help="Bin size used by PSMC (usually 100).",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=dgn.MU_HUMAN,
        help="Per-base mutation rate. Will be scaled by bin_size for the binned simulation.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=dgn.RHO_HUMAN,
        help="Per-base recombination rate. Will be scaled by bin_size for the binned simulation.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="hudson",
        help="msprime ancestry model name (passed to deepgene DataGenerator).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1,
        help="Sample size (passed to deepgene DataGenerator).",
    )

    out_group = parser.add_mutually_exclusive_group(required=True)
    out_group.add_argument(
        "--out",
        type=str,
        help="Output multi-fasta .psmcfa file (all replicates in one file).",
    )
    out_group.add_argument(
        "--out-dir",
        type=str,
        help="Output directory; writes one file per replicate (e.g. 0.psmcfa, 1.psmcfa, ...).",
    )

    parser.add_argument(
        "--line-length",
        type=int,
        default=60,
        help="FASTA line wrap length (PSMC examples typically use 60).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if they already exist.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.bin_size <= 0:
        raise ValueError("--bin-size must be > 0")
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be > 0")

    n_bins = int(math.ceil(args.sequence_length / args.bin_size))
    mu_bin = float(args.mu) * args.bin_size
    rho_bin = float(args.rho) * args.bin_size

    dem = _build_demography(
        args.demography, initial_size=args.initial_size, random_seed=args.random_seed
    )

    DG = dgn.DataGenerator(
        num_replicates=args.num_replicates,
        recombination_rate=rho_bin,
        mutation_rate=mu_bin,
        lengt=n_bins,
        demographic_events=dem,
        model=args.model,
        sample_size=args.sample_size,
        return_local_times=True,
        return_full_dist=False,
        genome_postproccessor=dgn.non_filter,
        times_postproccessor=dgn.non_filter,
    )

    if args.out is not None:
        out_path = Path(args.out)
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_path} (use --overwrite)")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for i, (mutations, _times) in enumerate(DG):
                seq = "".join(("T" if m == 0 else "K") for m in mutations)
                _write_record(f, str(i), seq, line_length=args.line_length)
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (mutations, _times) in enumerate(DG):
        p = out_dir / f"{i}.psmcfa"
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {p} (use --overwrite)")
        seq = "".join(("T" if m == 0 else "K") for m in mutations)
        with p.open("w", encoding="utf-8") as f:
            _write_record(f, str(i), seq, line_length=args.line_length)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

