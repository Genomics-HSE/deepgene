#!/usr/bin/env python3
"""
Convert consensus FASTQ (e.g. produced by `vcfutils.pl vcf2fq`) into PSMCFA.

This is a lightweight replacement for `fq2psmcfa`:
- We chunk the FASTQ sequence into bins of `--bin-size` bp (default 100).
- For each bin:
  - If there is at least one callable heterozygous IUPAC base (R,Y,S,W,K,M) with baseQ>=q -> 'K'
  - Else if there is at least one callable base with baseQ>=q -> 'T'
  - Else -> 'N'

Notes:
- This is designed to be compatible with the classic PSMC pipeline.
- It does not try to mimic every edge case of the original PSMC utilities, but keeps the core idea.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional, Tuple


HET_IUPAC = set("RYSWKM")  # heterozygous IUPAC codes used by vcfutils vcf2fq


def _iter_fastq_records(path: Path) -> Iterator[Tuple[str, str, str]]:
    """
    Yields (header, seq, qual). Header returned without leading '@'.

    Supports wrapped FASTQ, where sequence and/or quality can span multiple lines.
    This is important for `vcfutils.pl vcf2fq`, which often wraps long sequences.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            header_line = f.readline()
            if not header_line:
                break
            header_line = header_line.rstrip("\n")
            if not header_line.startswith("@"):
                raise ValueError(f"Invalid FASTQ: expected '@' header, got: {header_line[:50]!r}")
            header = header_line[1:].strip()

            # Read sequence lines until '+' separator
            seq_chunks = []
            while True:
                line = f.readline()
                if not line:
                    raise ValueError("Invalid FASTQ: unexpected EOF while reading sequence")
                line = line.rstrip("\n")
                if line.startswith("+"):
                    break
                seq_chunks.append(line.strip())
            seq = "".join(seq_chunks).upper()

            # Read quality lines until we have at least len(seq) chars
            qual_chunks = []
            qual_len = 0
            target = len(seq)
            while qual_len < target:
                ql = f.readline()
                if not ql:
                    raise ValueError("Invalid FASTQ: unexpected EOF while reading quality")
                ql = ql.rstrip("\n")
                ql_stripped = ql.strip()
                qual_chunks.append(ql_stripped)
                qual_len += len(ql_stripped)
            qual = "".join(qual_chunks)
            if len(qual) < len(seq):
                raise ValueError("Invalid FASTQ: quality shorter than sequence")
            if len(qual) > len(seq):
                qual = qual[: len(seq)]

            yield header, seq, qual


def _wrap_fasta(seq: str, line_len: int = 60) -> str:
    if line_len <= 0:
        return seq + "\n"
    return "\n".join(seq[i : i + line_len] for i in range(0, len(seq), line_len)) + "\n"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert consensus FASTQ to PSMCFA (fq2psmcfa replacement).")
    ap.add_argument("--fastq", required=True, help="Input FASTQ path")
    ap.add_argument("--out", required=True, help="Output .psmcfa path")
    ap.add_argument("--bin-size", type=int, default=100, help="Bin size in bp (default 100)")
    ap.add_argument("--q", type=int, default=20, help="Min base quality (Phred) to treat a position as callable (default 20)")
    ap.add_argument("--line-length", type=int, default=60, help="FASTA wrap length (default 60)")
    ap.add_argument("--record-index", type=int, default=0, help="Which FASTQ record to use (default 0)")
    ap.add_argument("--header", default=None, help="Override FASTA header (optional)")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    fq = Path(args.fastq)
    out = Path(args.out)

    if args.bin_size <= 0:
        raise ValueError("--bin-size must be > 0")
    if args.q < 0:
        raise ValueError("--q must be >= 0")

    rec = None
    for i, r in enumerate(_iter_fastq_records(fq)):
        if i == int(args.record_index):
            rec = r
            break
    if rec is None:
        raise IndexError(f"record_index={args.record_index} is out of range for {fq}")

    header, seq, qual = rec
    if len(seq) != len(qual):
        raise ValueError(f"FASTQ record has seq len {len(seq)} != qual len {len(qual)}")

    bin_size = int(args.bin_size)
    qmin = int(args.q)

    psmc_chars = []
    n = len(seq)
    for i in range(0, n, bin_size):
        sbin = seq[i : i + bin_size]
        qbin = qual[i : i + bin_size]

        has_callable = False
        has_het = False
        for b, qc in zip(sbin, qbin):
            phred = ord(qc) - 33
            if phred < qmin:
                continue
            if b == "N":
                continue
            has_callable = True
            if b in HET_IUPAC:
                has_het = True
                break

        if not has_callable:
            psmc_chars.append("N")
        elif has_het:
            psmc_chars.append("K")
        else:
            psmc_chars.append("T")

    psmc_seq = "".join(psmc_chars)
    out.parent.mkdir(parents=True, exist_ok=True)
    fasta_header = args.header if args.header is not None else header
    out.write_text(f">{fasta_header}\n{_wrap_fasta(psmc_seq, line_len=int(args.line_length))}", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

