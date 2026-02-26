#!/usr/bin/env python3
"""
Automatically select chromosomes to achieve target genome size (~30Mb for validation).

This script selects a combination of chromosomes from GRCh37 that gives a total size
close to the target size, suitable for validation on 1000 Genomes data.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

# GRCh37 chromosome sizes (in base pairs)
# Source: https://www.ncbi.nlm.nih.gov/grc/human/data?asm=GRCh37
CHROM_SIZES_GRCH37 = {
    "1": 249250621,
    "2": 243199373,
    "3": 198022430,
    "4": 191154276,
    "5": 180915260,
    "6": 171115067,
    "7": 159138663,
    "8": 146364022,
    "9": 141213431,
    "10": 135534747,
    "11": 135006516,
    "12": 133851895,
    "13": 115169878,
    "14": 107349540,
    "15": 102531392,
    "16": 90354753,
    "17": 81195210,
    "18": 78077248,
    "19": 59128983,
    "20": 63025520,
    "21": 48129895,
    "22": 51304566,
    "X": 155270560,
    "Y": 59373566,
    "MT": 16569,
}

# Also support chr-prefixed names
for k, v in list(CHROM_SIZES_GRCH37.items()):
    if k not in ("X", "Y", "MT"):
        CHROM_SIZES_GRCH37[f"chr{k}"] = v
    else:
        CHROM_SIZES_GRCH37[f"chr{k}"] = v


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Select chromosomes to achieve target genome size for 1000G validation."
    )
    ap.add_argument(
        "--target-size",
        type=int,
        default=30_000_000,
        help="Target genome size in bp (default: 30000000 = 30Mb)",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.2,
        help="Tolerance as fraction of target size (default: 0.2 = 20%%)",
    )
    ap.add_argument(
        "--prefer-small",
        action="store_true",
        help="Prefer smaller chromosomes (21, 22, etc.) over larger ones",
    )
    ap.add_argument(
        "--format",
        choices=["comma", "space", "list"],
        default="comma",
        help="Output format: comma-separated (default), space-separated, or one per line",
    )
    return ap.parse_args(argv)


def select_chromosomes(
    target_size: int,
    tolerance: float = 0.2,
    prefer_small: bool = False,
) -> list[str]:
    """
    Select chromosomes to achieve target size.
    
    Returns list of chromosome names (without 'chr' prefix by default).
    """
    # Sort chromosomes by size
    chrom_items = [
        (name, size)
        for name, size in CHROM_SIZES_GRCH37.items()
        if name not in ("X", "Y", "MT", "chrX", "chrY", "chrMT")
    ]
    
    if prefer_small:
        chrom_items.sort(key=lambda x: x[1])  # smallest first
    else:
        chrom_items.sort(key=lambda x: x[1])  # smallest first for better combinations
    
    min_size = target_size * (1 - tolerance)
    max_size = target_size * (1 + tolerance)
    
    # Try to find best combination using greedy approach
    best_combination = None
    best_diff = float("inf")
    
    # Try all combinations of 1-4 chromosomes
    from itertools import combinations
    
    for n in range(1, min(5, len(chrom_items) + 1)):
        for combo in combinations(chrom_items, n):
            total_size = sum(size for _, size in combo)
            
            diff = abs(total_size - target_size)
            if diff < best_diff:
                best_diff = diff
                best_combination = combo
                
                # If we're within tolerance, we can return early
                if min_size <= total_size <= max_size:
                    return [
                        name.replace("chr", "") if name.startswith("chr") else name
                        for name, _ in combo
                    ]
    
    # If we found a combination (even if outside tolerance), return it
    if best_combination:
        return [
            name.replace("chr", "") if name.startswith("chr") else name
            for name, _ in best_combination
        ]
    
    # Fallback: return smallest chromosome that's close
    best_chrom = None
    best_diff = float("inf")
    for chrom_name, chrom_size in chrom_items:
        diff = abs(chrom_size - target_size)
        if diff < best_diff:
            best_diff = diff
            best_chrom = chrom_name.replace("chr", "") if chrom_name.startswith("chr") else chrom_name
    return [best_chrom] if best_chrom else ["22"]


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    
    selected = select_chromosomes(
        target_size=int(args.target_size),
        tolerance=float(args.tolerance),
        prefer_small=bool(args.prefer_small),
    )
    
    if args.format == "comma":
        print(",".join(selected))
    elif args.format == "space":
        print(" ".join(selected))
    else:  # list
        for chrom in selected:
            print(chrom)
    
    # Print info to stderr
    total_size = sum(CHROM_SIZES_GRCH37.get(c, CHROM_SIZES_GRCH37.get(f"chr{c}", 0)) for c in selected)
    print(
        f"Selected chromosomes: {', '.join(selected)}",
        file=sys.stderr,
    )
    print(
        f"Total size: {total_size:,} bp ({total_size / 1_000_000:.2f} Mb)",
        file=sys.stderr,
    )
    print(
        f"Target: {args.target_size:,} bp ({args.target_size / 1_000_000:.2f} Mb)",
        file=sys.stderr,
    )
    print(
        f"Difference: {abs(total_size - args.target_size):,} bp ({abs(total_size - args.target_size) / args.target_size * 100:.1f}%%)",
        file=sys.stderr,
    )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
