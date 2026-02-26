#!/usr/bin/env python3
"""
Generate `runs_with_NN/` from `psmc_checker/runs/` by adding an NN curve.

For each synthetic run under runs/{const,hl,zigzag}:
- run NN inference (GRU checkpoint) on the corresponding `.psmcfa` (multi-fasta)
- aggregate probabilities into `model_dist` and convert to `Ne_nn` (mark_all.py formula)
- plot PSMC + TRUE + NN via psmc_checker/plot_results.py
- write per-run CSV with metrics:
  - PSMC vs TRUE: FMSE / RMSE / FMAE (relative errors like all_in_one.py)
  - NN   vs TRUE: FMSE / RMSE / FMAE
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "This script requires PyTorch installed and importable (torch). "
            "Your environment is missing a dependency required for NN inference."
        ) from e


class PredictorModule:  # thin wrapper to avoid importing deepgene.models.base_models (depends on extra pkgs)
    @staticmethod
    def build(*, state_dict: dict, n_class: int):
        import torch.nn as nn
        import torch.nn.functional as F

        class Predictor(nn.Module):
            def __init__(self, d_model: int, dropout: float, n_class_: int):
                super().__init__()
                self.dense1 = nn.Linear(d_model, d_model)
                self.dropout1 = nn.Dropout(dropout)
                self.dense2 = nn.Linear(d_model, d_model)
                self.dropout2 = nn.Dropout(dropout)
                self.dense3 = nn.Linear(d_model, n_class_)

            def forward(self, x):
                x = self.dropout1(F.relu(self.dense1(x)))
                x = self.dropout2(F.relu(self.dense2(x)))
                x = self.dense3(x)
                return x

        w = state_dict.get("predictor.dense1.weight")
        if w is None:
            raise ValueError("Cannot infer Predictor: missing predictor.dense1.weight in checkpoint")
        d_model = int(w.shape[0])
        return Predictor(d_model=d_model, dropout=0.1, n_class_=int(n_class))


class ConvEmbeddingModule:
    @staticmethod
    def build_from_state_dict(*, state_dict: dict, expected_dim: int):
        import torch.nn as nn

        conv_keys = [k for k in state_dict.keys() if k.startswith("embedding.conv_embeds.")]
        if not conv_keys:
            return None

        idxs = []
        for k in conv_keys:
            parts = k.split(".")
            if len(parts) >= 4 and parts[2].isdigit():
                idxs.append(int(parts[2]))
        n_layers = (max(idxs) + 1) if idxs else 1
        w0 = state_dict.get("embedding.conv_embeds.0.weight")
        if w0 is None:
            raise ValueError("Cannot infer ConvEmbedding without embedding.conv_embeds.0.weight")
        out_channels = int(w0.shape[0])
        in_channels = int(w0.shape[1])
        kernel_size = int(w0.shape[2])
        stride = 1
        if out_channels != expected_dim:
            raise ValueError(f"Checkpoint ConvEmbedding out_channels={out_channels} != expected_dim={expected_dim}")

        class ConvEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                conv_embeds = [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding="same",
                    )
                ]
                conv_embeds.extend(
                    [
                        nn.Conv1d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding="same",
                        )
                        for _ in range(n_layers - 1)
                    ]
                )
                self.conv_embeds = nn.ModuleList(conv_embeds)

            def forward(self, x):
                out = x.float().unsqueeze(1)  # (B,1,L)
                for layer in self.conv_embeds:
                    out = layer(out)
                return out.permute(0, 2, 1)  # (B,L,C)

        return ConvEmbedding()


class NoEmbeddingModule:
    @staticmethod
    def build():
        import torch.nn as nn

        class NoEmbedding(nn.Module):
            def forward(self, x):
                return x.unsqueeze(2).float()

        return NoEmbedding()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate runs_with_NN for synthetic PSMC runs.")

    ap.add_argument("--checkpoint", required=True, help="Path to GRU .ckpt (e.g. test_all_v1/model/gru_set3_ce.ckpt)")
    ap.add_argument(
        "--runs-dir",
        default=str(Path(__file__).resolve().parent / "runs"),
        help="Input directory with PSMC runs (default: psmc_checker/runs)",
    )
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "runs_with_NN"),
        help="Output directory (default: psmc_checker/runs_with_NN)",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs in out-dir")

    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--batch-size", type=int, default=1)

    # PSMC scaling / plotting controls (kept aligned with plot_results.py defaults)
    ap.add_argument("--bin-size", type=int, default=100)
    ap.add_argument("--generation-time", type=float, default=25.0)
    ap.add_argument("--mu", type=float, default=None, help="Override mu; default uses deepgene/data_gen_np MU_HUMAN")
    ap.add_argument("--psmc-time-mult", type=float, default=2.0)
    ap.add_argument("--psmc-size-mult", type=float, default=1.0)
    ap.add_argument("--y-max", type=float, default=20_000)

    return ap.parse_args(argv)


def _infer_true_demography_from_folder(folder: str) -> str:
    mapping = {
        "const": "get_const_demographcs",
        "hl": "get_humanlike",
        "zigzag": "get_zigzag",
    }
    if folder not in mapping:
        raise ValueError(f"Unknown run folder '{folder}', expected one of: {sorted(mapping.keys())}")
    return mapping[folder]


def _write_csv_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def _pick_device(device_arg: str):
    import torch

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _load_checkpoint_state_dict(ckpt_path: Path):
    import torch

    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt_obj.get("state_dict", ckpt_obj)
    if not isinstance(state_dict, dict):
        raise ValueError("Unexpected checkpoint format: expected dict or dict with 'state_dict'")
    return state_dict


def _infer_embedding(state_dict: dict, *, expected_dim: int):
    import torch.nn as nn

    keys = list(state_dict.keys())

    if any(k.startswith("embedding.weight") for k in keys):
        w = state_dict["embedding.weight"]
        num_embeddings = int(w.shape[0])
        embedding_dim = int(w.shape[1])
        if embedding_dim != expected_dim:
            raise ValueError(f"Checkpoint embedding_dim={embedding_dim} != expected_dim={expected_dim}")
        return nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    conv = ConvEmbeddingModule.build_from_state_dict(state_dict=state_dict, expected_dim=expected_dim)
    if conv is not None:
        return conv

    if expected_dim != 1:
        raise ValueError("No embedding weights found and expected_dim != 1 (incompatible checkpoint)")
    return NoEmbeddingModule.build()


def _infer_predictor(state_dict: dict, *, n_class: int):
    return PredictorModule.build(state_dict=state_dict, n_class=n_class)


def _build_gru_model(state_dict: dict):
    import re
    import torch.nn as nn

    w_ih = state_dict.get("gru.weight_ih_l0")
    if w_ih is None:
        raise ValueError("Not a GRU checkpoint (missing gru.weight_ih_l0)")
    input_size = int(w_ih.shape[1])
    hidden_size = int(w_ih.shape[0] // 3)

    layer_ids = set()
    for k in state_dict.keys():
        m = re.match(r"^gru\.weight_ih_l(\d+)$", k)
        if m:
            layer_ids.add(int(m.group(1)))
    num_layers = (max(layer_ids) + 1) if layer_ids else 1

    dense3_w = state_dict.get("predictor.dense3.weight")
    if dense3_w is None:
        raise ValueError("Missing predictor.dense3.weight")
    n_class = int(dense3_w.shape[0])

    embedding = _infer_embedding(state_dict, expected_dim=input_size)
    predictor = _infer_predictor(state_dict, n_class=n_class)

    class GruInfer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = embedding
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=0.1,
            )
            self.predictor = predictor

        def forward(self, x):
            x = self.embedding(x)
            out, _ = self.gru(x)
            out = self.predictor(out)
            return out

    model = GruInfer()
    return model


def _load_windows_from_psmcfa(psmcfa_path: Path) -> np.ndarray:
    from psmc_checker.psmcfa_helper import psmcfa_seq_to_binary, read_psmcfa_records

    windows = []
    expected_len = None
    for rec in read_psmcfa_records(psmcfa_path):
        x = psmcfa_seq_to_binary(rec.sequence)  # uint8, (L,)
        if expected_len is None:
            expected_len = int(x.shape[0])
        if int(x.shape[0]) != int(expected_len):
            # pad/trim to expected length to keep a rectangular batch
            w = np.zeros((int(expected_len),), dtype=np.uint8)
            n = min(int(expected_len), int(x.shape[0]))
            w[:n] = x[:n]
            x = w
        windows.append(x)
    if not windows:
        return np.zeros((0, 0), dtype=np.uint8)
    return np.stack(windows, axis=0)  # (n_records, L)


def _aggregate_model_dist_gru(
    *,
    checkpoint: Path,
    windows: np.ndarray,
    device: str,
    batch_size: int,
) -> list[float]:
    import torch
    import torch.nn as nn

    _require_torch()

    if windows.ndim != 2:
        raise ValueError(f"Expected windows 2D, got shape {windows.shape}")
    if int(batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")

    state_dict = _load_checkpoint_state_dict(checkpoint)
    model = _build_gru_model(state_dict)
    model.load_state_dict(state_dict, strict=True)

    dev = _pick_device(device)
    model.to(dev)
    model.eval()

    expects_long = isinstance(getattr(model, "embedding", None), nn.Embedding)

    priors = None
    total_positions = 0

    with torch.no_grad():
        for i in range(0, int(windows.shape[0]), int(batch_size)):
            batch = windows[i : i + int(batch_size)]
            xb = torch.from_numpy(batch)
            xb = xb.long() if expects_long else xb.float()
            xb = xb.to(dev)

            logits = model(xb)  # (B, L, C)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = torch.softmax(logits, dim=-1)

            probs_sum = probs.sum(dim=(0, 1)).detach().cpu().numpy().astype(np.float64)  # (C,)
            if priors is None:
                priors = probs_sum
            else:
                priors += probs_sum
            total_positions += int(probs.shape[0] * probs.shape[1])

    if priors is None or total_positions == 0:
        raise ValueError("No priors accumulated (empty windows?)")

    priors = priors / float(priors.sum())
    return [float(x) for x in priors.tolist()]


def _probability_to_N_mark_all(priors: list[float], limits: list[float]) -> list[float]:
    """
    Match mark_all.py behavior:
    - tau = limits[i+1] - limits[i]
    - use priors[j] for j in [0..len(tau)-1], ignoring any extra tail in priors
    """
    tau = np.asarray(limits[1:], dtype=float) - np.asarray(limits[:-1], dtype=float)
    pri = np.asarray(priors, dtype=float)
    if pri.shape[0] < tau.shape[0]:
        raise ValueError(f"Expected priors length >= {tau.shape[0]}, got {pri.shape[0]}")

    l = 1.0
    out: list[float] = []
    eps = 1e-12
    for j in range(int(tau.shape[0])):
        pj = float(pri[j])
        # guard against log(<=0)
        val = 1.0 - pj / l
        val = max(min(val, 1.0 - eps), eps)
        Nj = float(np.round(float(tau[j]) / (-2.0 * math.log(val))))
        out.append(Nj)
        l *= float(math.exp(-float(tau[j]) / (2.0 * Nj)))
    return out


def _relative_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, int]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
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


def _true_ne_at_times(*, limits_x: list[float], true_ne: np.ndarray, times: list[float]) -> np.ndarray:
    x = np.asarray(limits_x, dtype=float)
    y = np.asarray(true_ne, dtype=float).reshape(-1)
    t = np.asarray(times, dtype=float)
    idx = np.searchsorted(x, t, side="right") - 1
    idx = np.clip(idx, 0, len(y) - 1)
    return y[idx]


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ckpt = Path(args.checkpoint).expanduser().resolve()

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs dir not found: {runs_dir}")

    # Use existing plotter + demography helpers
    from psmc_checker import plot_results

    mu_used = float(args.mu) if args.mu is not None else float(getattr(plot_results.dgn, "MU_HUMAN"))
    limits_full = list(getattr(plot_results.dgn, "limits"))
    # Для построения графиков используем полные limits, для метрик - обрезанные
    limits = limits_full  # Полные limits для plot_results
    limits_x = [float(t) * float(args.generation_time) for t in limits]
    # Ограниченные limits для вычисления метрик (первые 30 интервалов)
    limits_30 = limits_full[:31]  # 31 элемент для 30 интервалов
    limits_x_30 = [float(t) * float(args.generation_time) for t in limits_30]

    # Synthetic folders (have TRUE demography)
    psmc_files: list[Path] = []
    for dem_folder in ("const", "hl", "zigzag"):
        psmc_files.extend(sorted((runs_dir / dem_folder).glob("*.psmc")))

    # 1000G (no TRUE demography): expect `.psmc` already present in out_dir/1000g
    psmc_files_1000g = sorted((out_dir / "1000g").glob("*.psmc"))

    if not psmc_files and not psmc_files_1000g:
        print(f"No .psmc files found under {runs_dir}/(const|hl|zigzag) and {out_dir}/1000g", file=sys.stderr)
        return 1

    processed = 0
    failed = 0

    def process_one(psmc_path: Path, *, dem_folder: str, true_demo: Optional[str]) -> None:
        nonlocal processed, failed

        # For synthetic: .psmcfa sits next to .psmc in runs_dir/{dem}/
        # For 1000g: prefer `.psmcfa` next to `.psmc` in out_dir/1000g, else fallback to runs_dir/1000g/
        psmcfa_path = psmc_path.with_suffix(".psmcfa")
        if not psmcfa_path.exists():
            candidate = runs_dir / dem_folder / psmc_path.with_suffix(".psmcfa").name
            if candidate.exists():
                psmcfa_path = candidate

        if not psmcfa_path.exists():
            print(f"[fail] missing .psmcfa for {psmc_path}", file=sys.stderr)
            failed += 1
            return

        out_dem_dir = out_dir / dem_folder
        out_dem_dir.mkdir(parents=True, exist_ok=True)

        stem = psmc_path.stem
        out_png = out_dem_dir / f"{stem}.png"
        out_csv = out_dem_dir / f"{stem}.csv"
        out_nn_priors = out_dem_dir / f"{stem}.nn_priors.json"
        out_nn_ne = out_dem_dir / f"{stem}.nn_ne.json"

        if out_png.exists() and not args.overwrite:
            return

        # Copy inputs for traceability
        dst_psmc = out_dem_dir / psmc_path.name
        dst_psmcfa = out_dem_dir / psmcfa_path.name
        if psmc_path.resolve() != dst_psmc.resolve():
            shutil.copy2(psmc_path, dst_psmc)
        if psmcfa_path.resolve() != dst_psmcfa.resolve():
            shutil.copy2(psmcfa_path, dst_psmcfa)

        try:
            windows = _load_windows_from_psmcfa(psmcfa_path)
            if windows.size == 0:
                raise ValueError(f"Empty windows from {psmcfa_path}")

            priors = _aggregate_model_dist_gru(
                checkpoint=ckpt,
                windows=windows,
                device=str(args.device),
                batch_size=int(args.batch_size),
            )

            out_nn_priors.write_text(
                json.dumps(
                    {
                        "checkpoint": str(ckpt),
                        "n_records": int(windows.shape[0]),
                        "window_len": int(windows.shape[1]),
                        "n_class": int(len(priors)),
                        "priors": priors,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            nn_ne_full = _probability_to_N_mark_all(priors, limits)
            # Для построения графиков используем полные nn_ne
            out_nn_ne.write_text(json.dumps(nn_ne_full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            # Для метрик обрезаем до первых 30 интервалов
            nn_ne = nn_ne_full[:30]

            # Plot
            pr_argv = [
                "--psmc",
                str(dst_psmc),
                "--out",
                str(out_png),
                "--nn-ne-json",
                str(out_nn_ne),
                "--bin-size",
                str(int(args.bin_size)),
                "--mu",
                str(float(mu_used)),
                "--generation-time",
                str(float(args.generation_time)),
                "--psmc-time-mult",
                str(float(args.psmc_time_mult)),
                "--psmc-size-mult",
                str(float(args.psmc_size_mult)),
                "--y-max",
                str(float(args.y_max)),
            ]
            if true_demo is not None:
                pr_argv += ["--true-demography", true_demo]
            plot_results.main(pr_argv)

            # Metrics
            psmc_times, psmc_sizes, n0 = plot_results.psmc_parse_times_sizes(
                dst_psmc,
                bin_size=int(args.bin_size),
                mutation_rate=float(mu_used),
                generation_time=float(args.generation_time),
                N0_override=None,
                time_mult=float(args.psmc_time_mult),
                size_mult=float(args.psmc_size_mult),
            )

            if true_demo is not None:
                dem = plot_results.build_demography(true_demo, initial_size=10_000, random_seed=42)
                true_ne_full = np.asarray(plot_results.demography_to_Ne(dem, limits), dtype=float).reshape(-1)

                # NN vs TRUE on interval grid (только первые 30 интервалов)
                true_ne_intervals_full = true_ne_full[:-1]
                true_ne_intervals = true_ne_intervals_full[:30]  # Обрезаем до 30 интервалов
                nn_fmse, nn_rmse, nn_fmae, nn_n = _relative_metrics(true_ne_intervals, np.asarray(nn_ne, dtype=float))

                # PSMC vs TRUE on PSMC timepoints
                true_at_psmc = _true_ne_at_times(limits_x=limits_x, true_ne=true_ne_full, times=psmc_times)
                psmc_fmse, psmc_rmse, psmc_fmae, psmc_n = _relative_metrics(true_at_psmc, np.asarray(psmc_sizes, dtype=float))

                _write_csv_row(
                    out_csv,
                    {
                        "psmc_path": str(dst_psmc),
                        "psmcfa_path": str(dst_psmcfa),
                        "png_path": str(out_png),
                        "checkpoint": str(ckpt),
                        "true_demography": true_demo,
                        "bin_size": int(args.bin_size),
                        "mu": float(mu_used),
                        "generation_time": float(args.generation_time),
                        "N0": float(n0),
                        "n_records": int(windows.shape[0]),
                        "window_len": int(windows.shape[1]),
                        "psmc_n_points": int(psmc_n),
                        "psmc_FMSE": psmc_fmse,
                        "psmc_RMSE": psmc_rmse,
                        "psmc_FMAE": psmc_fmae,
                        "nn_n_points": int(nn_n),
                        "nn_FMSE": nn_fmse,
                        "nn_RMSE": nn_rmse,
                        "nn_FMAE": nn_fmae,
                    },
                )
            else:
                # 1000G: no TRUE, compute NN vs PSMC (PSMC treated as ground truth) on PSMC timepoints.
                nn_full = np.asarray(list(nn_ne) + [nn_ne[-1]], dtype=float)  # align with step-post plotting
                nn_at_psmc = _true_ne_at_times(limits_x=limits_x, true_ne=nn_full, times=psmc_times)
                fmse, rmse, fmae, n_pts = _relative_metrics(np.asarray(psmc_sizes, dtype=float), nn_at_psmc)

                _write_csv_row(
                    out_csv,
                    {
                        "psmc_path": str(dst_psmc),
                        "psmcfa_path": str(dst_psmcfa),
                        "png_path": str(out_png),
                        "checkpoint": str(ckpt),
                        "bin_size": int(args.bin_size),
                        "mu": float(mu_used),
                        "generation_time": float(args.generation_time),
                        "N0": float(n0),
                        "n_records": int(windows.shape[0]),
                        "window_len": int(windows.shape[1]),
                        "n_points": int(n_pts),
                        "nn_vs_psmc_FMSE": fmse,
                        "nn_vs_psmc_RMSE": rmse,
                        "nn_vs_psmc_FMAE": fmae,
                    },
                )

            processed += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[fail] {psmc_path}: {type(e).__name__}: {e}", file=sys.stderr)

    # Изменяем out_dir для сохранения в new_res_30/psmc_synthetic/
    original_out_dir = out_dir
    out_dir = Path("new_res_30") / "psmc_synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process synthetic
    for psmc_path in psmc_files:
        dem_folder = psmc_path.parent.name
        process_one(psmc_path, dem_folder=dem_folder, true_demo=_infer_true_demography_from_folder(dem_folder))

    # Process 1000g (if present)
    for psmc_path in psmc_files_1000g:
        process_one(psmc_path, dem_folder="1000g", true_demo=None)

    print(
        f"Done. processed={processed} failed={failed} out_dir={out_dir}",
        file=sys.stderr if failed else sys.stdout,
    )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

