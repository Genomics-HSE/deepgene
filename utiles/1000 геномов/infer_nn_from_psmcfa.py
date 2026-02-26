#!/usr/bin/env python3
"""
Run NN inference on PSMCFA-derived windows and save aggregated outputs.

Inputs:
- Either --windows (npz produced by psmcfa_to_nn_windows.py)
- Or --psmcfa (will be converted to windows on the fly)

Outputs:
- pred_probs.npy  (float32, shape=(n_windows, window_len, n_class)) by default
- nn_priors.json  (aggregated probability mass per class, length=n_class)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Infer deepgene model on .psmcfa/windows.")
    ap.add_argument("--checkpoint", required=True, help="Path to .ckpt (Lightning checkpoint)")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--windows", help="Path to nn_windows.npz (contains 'windows')")
    src.add_argument("--psmcfa", help="Path to .psmcfa (will be converted to windows)")

    ap.add_argument("--outdir", required=True, help="Output directory")

    ap.add_argument("--model-type", choices=["auto", "gru", "lstm", "reformer"], default="auto")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save", choices=["probs", "logits"], default="probs")

    # On-the-fly window params (only used with --psmcfa)
    ap.add_argument("--record-index", type=int, default=0)
    ap.add_argument("--bin-size", type=int, default=100)
    ap.add_argument("--chrom", default=None)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--window-len", type=int, default=30000)
    ap.add_argument("--step", type=int, default=30000)
    ap.add_argument("--pad", action="store_true")

    # Reformer-only (gin config path); used only if model type is reformer
    ap.add_argument("--reformer-gin", default=None, help="Path to configs/model_reformer.gin (optional)")
    ap.add_argument("--loss-function", default="CrossEntropyLoss", help="Used only for gin placeholders (default CrossEntropyLoss)")
    ap.add_argument("--n-class", type=int, default=32, help="Number of classes (default 32)")
    return ap.parse_args(argv)


def _pick_device(device_arg: str):
    import torch

    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # auto
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _load_checkpoint_state_dict(ckpt_path: Path):
    import torch

    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt_obj.get("state_dict", ckpt_obj)
    if not isinstance(state_dict, dict):
        raise ValueError("Unexpected checkpoint format: expected dict or dict with 'state_dict'")
    return state_dict


def _detect_model_type(state_dict: dict, forced: str) -> str:
    if forced != "auto":
        return forced
    keys = list(state_dict.keys())
    if any(k.startswith("gru.") for k in keys):
        return "gru"
    if any(k.startswith("lstm.") for k in keys):
        return "lstm"
    if any("reformer" in k.lower() for k in keys) or any(k.startswith("config.") for k in keys):
        return "reformer"
    raise ValueError("Cannot auto-detect model type from checkpoint state_dict keys")


def _infer_embedding(state_dict: dict, *, expected_dim: int):
    import torch.nn as nn
    from deepgene.models.base_models import ConvEmbedding, NoEmbedding

    keys = list(state_dict.keys())

    if any(k.startswith("embedding.weight") for k in keys):
        # nn.Embedding: weight shape = (num_embeddings, embedding_dim)
        w = state_dict["embedding.weight"]
        num_embeddings = int(w.shape[0])
        embedding_dim = int(w.shape[1])
        if embedding_dim != expected_dim:
            raise ValueError(f"Checkpoint embedding_dim={embedding_dim} != expected_dim={expected_dim}")
        return nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    conv_keys = [k for k in keys if k.startswith("embedding.conv_embeds.")]
    if conv_keys:
        # Determine n_layers by the maximum conv index.
        idxs = []
        for k in conv_keys:
            parts = k.split(".")
            # embedding.conv_embeds.{i}.weight
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
        return ConvEmbedding(
            n_layers=n_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

    # NoEmbedding has no parameters in state_dict.
    if expected_dim != 1:
        raise ValueError(
            "No embedding weights found in checkpoint, but expected_dim != 1. "
            "This likely means embedding is missing or incompatible."
        )
    return NoEmbedding()


def _infer_predictor(state_dict: dict, *, n_class: int):
    from deepgene.models.base_models import Predictor

    w = state_dict.get("predictor.dense1.weight")
    if w is None:
        raise ValueError("Cannot infer Predictor: missing predictor.dense1.weight in checkpoint")
    d_model = int(w.shape[0])

    # dropout isn't in state_dict; use a safe default
    dropout = 0.1
    return Predictor(d_model=d_model, dropout=dropout, n_class=n_class)


def _build_gru_model(state_dict: dict):
    import re
    from deepgene.models.gru import GruLabeler

    w_ih = state_dict.get("gru.weight_ih_l0")
    if w_ih is None:
        raise ValueError("Not a GRU checkpoint (missing gru.weight_ih_l0)")
    input_size = int(w_ih.shape[1])
    hidden_size = int(w_ih.shape[0] // 3)

    # infer num_layers from forward-direction layer keys (exclude *_reverse)
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

    model = GruLabeler(
        embedding=embedding,
        n_class=n_class,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        predictor=predictor,
        loss="CrossEntropyLoss",
    )
    return model


def _build_lstm_model(state_dict: dict):
    import re
    import torch.nn as nn

    class LstmInferModel(nn.Module):
        def __init__(self, embedding, lstm, predictor):
            super().__init__()
            self.embedding = embedding
            self.lstm = lstm
            self.predictor = predictor

        def forward(self, x):
            x = self.embedding(x)
            out, _ = self.lstm(x)
            out = self.predictor(out)
            return out

    w_ih = state_dict.get("lstm.weight_ih_l0")
    if w_ih is None:
        raise ValueError("Not an LSTM checkpoint (missing lstm.weight_ih_l0)")
    input_size = int(w_ih.shape[1])
    hidden_size = int(w_ih.shape[0] // 4)

    layer_ids = set()
    for k in state_dict.keys():
        m = re.match(r"^lstm\.weight_ih_l(\d+)$", k)
        if m:
            layer_ids.add(int(m.group(1)))
    num_layers = (max(layer_ids) + 1) if layer_ids else 1

    bidirectional = any(k.startswith("lstm.weight_ih_l0_reverse") for k in state_dict.keys())

    dense3_w = state_dict.get("predictor.dense3.weight")
    if dense3_w is None:
        raise ValueError("Missing predictor.dense3.weight")
    n_class = int(dense3_w.shape[0])

    embedding = _infer_embedding(state_dict, expected_dim=input_size)
    predictor = _infer_predictor(state_dict, n_class=n_class)

    dropout = 0.1 if num_layers > 1 else 0.0
    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    model = LstmInferModel(embedding=embedding, lstm=lstm, predictor=predictor)
    return model


def _build_reformer_model(state_dict: dict, *, gin_path: Optional[str], n_class: int, loss_function: str):
    import gin
    import deepgene.models

    if gin_path is None:
        # default to repo config if present
        here = Path(__file__).resolve()
        repo_root = here.parent.parent
        candidate = repo_root / "configs" / "model_reformer.gin"
        gin_path = str(candidate) if candidate.exists() else None
    if gin_path is None:
        raise ValueError("Reformer inference requires --reformer-gin (cannot find default configs/model_reformer.gin)")

    gin.clear_config()
    gin.parse_config_files_and_bindings(
        [gin_path],
        [
            "model_name='infer'",
            f"loss_function='{loss_function}'",
            f"n_class={int(n_class)}",
        ],
    )

    model = deepgene.models.ReformerLabeler()

    # Lightning checkpoints can include extra keys; align with current model definition.
    model_keys = set(model.state_dict().keys())
    extra = [k for k in state_dict.keys() if k not in model_keys]
    if extra:
        for k in extra:
            del state_dict[k]

    model.load_state_dict(state_dict, strict=True)
    return model


def _load_windows_from_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if "windows" not in data:
        raise ValueError(f"{path} does not contain 'windows' array")
    x = data["windows"]
    if x.ndim != 2:
        raise ValueError(f"Expected windows to be 2D, got shape {x.shape}")
    return x


def _make_windows_from_psmcfa(args: argparse.Namespace) -> np.ndarray:
    # Local import to keep startup light if not needed
    # Ensure repo root is importable when running this script by path.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from psmc_checker.psmcfa_helper import read_psmcfa_records, psmcfa_seq_to_binary

    p = Path(args.psmcfa)
    rec = None
    for i, r in enumerate(read_psmcfa_records(p)):
        if i == int(args.record_index):
            rec = r
            break
    if rec is None:
        raise IndexError(f"record_index={args.record_index} is out of range for {p}")

    x = psmcfa_seq_to_binary(rec.sequence)  # (n_bins,)
    n_bins = int(x.shape[0])

    windows = []
    for start in range(0, n_bins, int(args.step)):
        end = start + int(args.window_len)
        if end <= n_bins:
            windows.append(x[start:end])
        else:
            if not args.pad:
                break
            w = np.zeros((int(args.window_len),), dtype=np.uint8)
            w[: max(0, n_bins - start)] = x[start:n_bins]
            windows.append(w)
            break
    if not windows:
        return np.zeros((0, int(args.window_len)), dtype=np.uint8)
    return np.stack(windows, axis=0)


def main(argv: Optional[list[str]] = None) -> int:
    # Ensure repo root is importable when running this script by path.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    
    args = parse_args(argv)

    ckpt_path = Path(args.checkpoint)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    state_dict = _load_checkpoint_state_dict(ckpt_path)
    model_type = _detect_model_type(state_dict, forced=args.model_type)

    if args.windows:
        windows = _load_windows_from_npz(Path(args.windows))
    else:
        windows = _make_windows_from_psmcfa(args)

    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")

    device = _pick_device(args.device)

    # Build model and load weights
    if model_type == "gru":
        model = _build_gru_model(state_dict)
    elif model_type == "lstm":
        model = _build_lstm_model(state_dict)
    elif model_type == "reformer":
        model = _build_reformer_model(
            state_dict,
            gin_path=args.reformer_gin,
            n_class=int(args.n_class),
            loss_function=str(args.loss_function),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    import torch

    model.to(device)
    model.eval()

    # Decide input dtype based on embedding kind
    # - nn.Embedding expects Long tensor
    expects_long = False
    if hasattr(model, "embedding"):
        emb = getattr(model, "embedding")
        import torch.nn as nn

        expects_long = isinstance(emb, nn.Embedding)

    # Inference
    n_windows, window_len = int(windows.shape[0]), int(windows.shape[1])
    outputs = []
    priors = None  # aggregated per-class probability mass
    total_positions = 0

    with torch.no_grad():
        for i in range(0, n_windows, int(args.batch_size)):
            batch = windows[i : i + int(args.batch_size)]
            xb = torch.from_numpy(batch)
            if expects_long:
                xb = xb.long()
            else:
                xb = xb.float()
            xb = xb.to(device)

            logits = model(xb)  # (B, L, C)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            if args.save == "logits":
                out = logits.detach().cpu().float().numpy()
            else:
                probs = torch.softmax(logits, dim=-1)
                out = probs.detach().cpu().float().numpy()

                # accumulate priors
                probs_sum = probs.sum(dim=(0, 1)).detach().cpu().numpy()  # (C,)
                if priors is None:
                    priors = probs_sum.astype(np.float64)
                else:
                    priors += probs_sum.astype(np.float64)
                total_positions += int(probs.shape[0] * probs.shape[1])

            outputs.append(out)

    pred = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, window_len, int(args.n_class)), dtype=np.float32)

    if args.save == "logits":
        np.save(outdir / "pred_logits.npy", pred.astype(np.float32))
        # priors from logits isn't well-defined; still compute via softmax for convenience
        p = torch.softmax(torch.from_numpy(pred), dim=-1).sum(dim=(0, 1)).numpy().astype(np.float64)
        priors = p
        total_positions = int(pred.shape[0] * pred.shape[1])
    else:
        np.save(outdir / "pred_probs.npy", pred.astype(np.float32))

    if priors is None or total_positions == 0:
        priors_norm = []
    else:
        priors_norm = (priors / float(priors.sum())).tolist()

    (outdir / "nn_priors.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "checkpoint": str(ckpt_path),
                "n_windows": n_windows,
                "window_len": window_len,
                "n_class": int(pred.shape[-1]) if pred.ndim == 3 else int(args.n_class),
                "total_positions": total_positions,
                "priors": priors_norm,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

