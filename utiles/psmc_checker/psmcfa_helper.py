"""
Helper utilities to load PSMC input format (.psmcfa) into a NN-friendly representation.

PSMCFA is a FASTA-like file where sequence characters encode heterozygosity per bin:
- 'K' usually marks a heterozygous bin (=> 1)
- 'T' usually marks a homozygous bin (=> 0)
Real data may also contain 'N' (missing) and sometimes 'A'/'C'/'G' depending on preprocessing.

This module focuses on:
- reading records from multi-fasta `.psmcfa`
- converting sequences to a binary vector (0/1)
- slicing into fixed-length windows for NN inference/training
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PsmcfaRecord:
    header: str
    sequence: str


def read_psmcfa_records(path: str | Path) -> Iterator[PsmcfaRecord]:
    """
    Reads a .psmcfa file (possibly multi-fasta) and yields records.

    Notes:
    - Header is returned without the leading '>'.
    - Sequence is returned as a single uppercased string with all whitespace removed.
    """
    p = Path(path)
    header: Optional[str] = None
    chunks: list[str] = []

    with p.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield PsmcfaRecord(header=header, sequence="".join(chunks).upper())
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)

    if header is not None:
        yield PsmcfaRecord(header=header, sequence="".join(chunks).upper())


def psmcfa_seq_to_binary(
    seq: str,
    *,
    one_chars: Tuple[str, ...] = ("K", "A"),
    zero_chars: Tuple[str, ...] = ("T", "N"),
    other_to_zero: bool = True,
) -> np.ndarray:
    """
    Converts a PSMCFA sequence into a 0/1 numpy vector.

    - Characters in one_chars map to 1
    - Characters in zero_chars map to 0
    - Other characters map to 0 if other_to_zero=True, otherwise raises ValueError
    """
    one = set(one_chars)
    zero = set(zero_chars)

    out = np.zeros(len(seq), dtype=np.uint8)
    for i, ch in enumerate(seq):
        if ch in one:
            out[i] = 1
        elif ch in zero:
            out[i] = 0
        elif other_to_zero:
            out[i] = 0
        else:
            raise ValueError(f"Unexpected character in psmcfa sequence: {ch!r}")
    return out


def load_psmcfa_as_binary(
    path: str | Path,
    *,
    record_index: int = 0,
    one_chars: Tuple[str, ...] = ("K", "A"),
    zero_chars: Tuple[str, ...] = ("T", "N"),
) -> np.ndarray:
    """
    Convenience function: load one record from .psmcfa and return its binary vector.
    """
    for i, rec in enumerate(read_psmcfa_records(path)):
        if i == record_index:
            return psmcfa_seq_to_binary(rec.sequence, one_chars=one_chars, zero_chars=zero_chars)
    raise IndexError(f"record_index={record_index} is out of range for {path}")


def iter_windows(
    x: np.ndarray,
    *,
    window_len: int,
    step: Optional[int] = None,
    pad: bool = False,
    pad_value: int = 0,
) -> Iterator[np.ndarray]:
    """
    Iterate over 1D windows of fixed length.

    - If pad=False: drops the tail that doesn't fit a full window.
    - If pad=True: yields the last window padded to window_len.
    """
    if x.ndim != 1:
        raise ValueError("iter_windows expects a 1D array")
    if window_len <= 0:
        raise ValueError("window_len must be > 0")
    if step is None:
        step = window_len
    if step <= 0:
        raise ValueError("step must be > 0")

    n = int(x.shape[0])
    i = 0
    while i < n:
        j = i + window_len
        if j <= n:
            yield x[i:j]
        elif pad:
            out = np.full((window_len,), pad_value, dtype=x.dtype)
            out[: n - i] = x[i:n]
            yield out
            break
        else:
            break
        i += step


def to_torch_tensor(x: np.ndarray):
    """
    Optional helper: convert numpy array to torch tensor (if torch is installed).
    """
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("torch is not available in this environment") from e
    return torch.tensor(x)

