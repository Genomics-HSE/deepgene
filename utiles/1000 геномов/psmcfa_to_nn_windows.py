#!/usr/bin/env python3
"""
Convert a PSMC input file (.psmcfa) to NN-friendly fixed-length windows.

Output:
- .npz with `windows` array (uint8, shape=(n_windows, window_len))
- .tsv manifest with window coordinates
- .json meta with parameters
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure repo root is importable when running this script by path (Python sets sys.path[0] to this directory).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psmc_checker.psmcfa_helper import read_psmcfa_records, psmcfa_seq_to_binary  # noqa: E402


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert .psmcfa to NN windows (0/1).")
    
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--psmcfa", help="Input .psmcfa path (single file)")
    input_group.add_argument("--combine-psmcfa", nargs="+", help="Multiple .psmcfa files to combine (space-separated)")
    
    ap.add_argument("--record-index", type=int, default=0, help="Which FASTA record to use (default 0)")

    ap.add_argument("--bin-size", type=int, default=100, help="Bin size in bp used to create the .psmcfa (default 100)")
    ap.add_argument("--chrom", default=None, help="Chromosome name for metadata/manifest (optional, or comma-separated for --combine-psmcfa)")
    ap.add_argument("--sample", default=None, help="Sample ID for metadata (optional)")

    ap.add_argument("--window-len", type=int, default=30000, help="Window length in bins (default 30000)")
    ap.add_argument("--step", type=int, default=30000, help="Step in bins (default 30000)")
    ap.add_argument("--pad", action="store_true", help="Pad the last window instead of dropping the tail")

    ap.add_argument("--out", required=True, help="Output .npz path (will contain 'windows')")
    ap.add_argument("--manifest", required=True, help="Output TSV manifest path")
    ap.add_argument("--meta", required=True, help="Output JSON metadata path")
    return ap.parse_args(argv)


def _select_record(psmcfa_path: Path, record_index: int):
    for i, rec in enumerate(read_psmcfa_records(psmcfa_path)):
        if i == record_index:
            return rec
    raise IndexError(f"record_index={record_index} is out of range for {psmcfa_path}")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    out_path = Path(args.out)
    manifest_path = Path(args.manifest)
    meta_path = Path(args.meta)

    if args.window_len <= 0:
        raise ValueError("--window-len must be > 0")
    if args.step <= 0:
        raise ValueError("--step must be > 0")
    if args.bin_size <= 0:
        raise ValueError("--bin-size must be > 0")

    # Handle single file or multiple files
    if args.combine_psmcfa is not None:
        # Combine multiple .psmcfa files
        psmcfa_paths = [Path(p) for p in args.combine_psmcfa]
        
        # Parse chromosome names
        if args.chrom and "," in args.chrom:
            chroms = [c.strip() for c in args.chrom.split(",")]
            if len(chroms) != len(psmcfa_paths):
                # If not enough chroms provided, pad with None or repeat last
                if len(chroms) == 1:
                    chroms = [chroms[0]] * len(psmcfa_paths)
                else:
                    chroms.extend([None] * (len(psmcfa_paths) - len(chroms)))
        elif args.chrom:
            chroms = [args.chrom] * len(psmcfa_paths)
        else:
            chroms = [None] * len(psmcfa_paths)
        
        all_windows: list[np.ndarray] = []
        all_rows: list[tuple] = []
        all_meta_chroms = []
        total_windows = 0
        
        for psmcfa_path, chrom in zip(psmcfa_paths, chroms):
            if not psmcfa_path.exists():
                raise FileNotFoundError(f"PSMCFA file not found: {psmcfa_path}")
            
            rec = _select_record(psmcfa_path, args.record_index)
            x = psmcfa_seq_to_binary(rec.sequence)  # uint8, shape=(n_bins,)
            
            n_bins = int(x.shape[0])
            starts = list(range(0, n_bins, int(args.step)))
            
            chrom_windows: list[np.ndarray] = []
            chrom_rows: list[tuple] = []
            
            for start_bin in starts:
                end_bin = start_bin + int(args.window_len)
                if end_bin <= n_bins:
                    w = x[start_bin:end_bin]
                else:
                    if not args.pad:
                        break
                    w = np.zeros((int(args.window_len),), dtype=np.uint8)
                    w[: max(0, n_bins - start_bin)] = x[start_bin:n_bins]
                    end_bin = min(end_bin, n_bins)
                
                chrom_windows.append(w)
                
                start_bp = start_bin * int(args.bin_size)
                end_bp = end_bin * int(args.bin_size)
                chrom_rows.append(
                    (
                        total_windows + len(chrom_windows) - 1,
                        chrom or str(psmcfa_path.stem),
                        start_bin,
                        end_bin,
                        start_bp,
                        end_bp,
                    )
                )
            
            all_windows.extend(chrom_windows)
            all_rows.extend(chrom_rows)
            total_windows += len(chrom_windows)
            
            all_meta_chroms.append({
                "chrom": chrom or str(psmcfa_path.stem),
                "psmcfa": str(psmcfa_path),
                "header": rec.header,
                "n_bins": int(n_bins),
                "n_windows": len(chrom_windows),
            })
        
        windows_arr = np.stack(all_windows, axis=0) if all_windows else np.zeros((0, int(args.window_len)), dtype=np.uint8)
        
        meta = {
            "sample": args.sample,
            "chrom": args.chrom if args.chrom and "," not in args.chrom else "multiple",
            "chromosomes": all_meta_chroms,
            "record_index": int(args.record_index),
            "bin_size": int(args.bin_size),
            "window_len": int(args.window_len),
            "step": int(args.step),
            "pad": bool(args.pad),
            "n_windows": int(windows_arr.shape[0]),
            "combined": True,
        }
    else:
        # Single file processing (original behavior)
        psmcfa_path = Path(args.psmcfa)
        if not psmcfa_path.exists():
            raise FileNotFoundError(f"PSMCFA file not found: {psmcfa_path}")
        
        rec = _select_record(psmcfa_path, args.record_index)
        x = psmcfa_seq_to_binary(rec.sequence)  # uint8, shape=(n_bins,)

        n_bins = int(x.shape[0])
        starts = list(range(0, n_bins, int(args.step)))

        windows: list[np.ndarray] = []
        rows: list[tuple] = []
        for w_idx, start_bin in enumerate(starts):
            end_bin = start_bin + int(args.window_len)
            if end_bin <= n_bins:
                w = x[start_bin:end_bin]
            else:
                if not args.pad:
                    break
                w = np.zeros((int(args.window_len),), dtype=np.uint8)
                w[: max(0, n_bins - start_bin)] = x[start_bin:n_bins]
                end_bin = min(end_bin, n_bins)

            windows.append(w)

            start_bp = start_bin * int(args.bin_size)
            end_bp = end_bin * int(args.bin_size)
            rows.append(
                (
                    w_idx,
                    args.chrom or "",
                    start_bin,
                    end_bin,
                    start_bp,
                    end_bp,
                )
            )

        windows_arr = np.stack(windows, axis=0) if windows else np.zeros((0, int(args.window_len)), dtype=np.uint8)
        all_rows = rows

        meta = {
            "sample": args.sample,
            "chrom": args.chrom,
            "psmcfa": str(psmcfa_path),
            "record_index": int(args.record_index),
            "header": rec.header,
            "bin_size": int(args.bin_size),
            "n_bins": int(n_bins),
            "window_len": int(args.window_len),
            "step": int(args.step),
            "pad": bool(args.pad),
            "n_windows": int(windows_arr.shape[0]),
            "combined": False,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, windows=windows_arr)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("window_index\tchrom\tstart_bin\tend_bin\tstart_bp\tend_bp\n")
        for row in all_rows:
            f.write("\t".join(map(str, row)) + "\n")

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

