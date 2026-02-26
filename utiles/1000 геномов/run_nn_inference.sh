#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run NN inference on prepared NN windows (or directly from .psmcfa).

Examples:
  bash "1000 геномов/run_nn_inference.sh" \
    --checkpoint /path/to/gru_const_ce.ckpt \
    --windows /path/to/nn_windows.npz \
    --outdir /path/to/nn_infer

  bash "1000 геномов/run_nn_inference.sh" \
    --checkpoint /path/to/gru_const_ce.ckpt \
    --psmcfa /path/to/input.psmcfa \
    --outdir /path/to/nn_infer

Notes:
  - Model type is auto-detected by default (gru/lstm/reformer).
  - For reformer, you may need --reformer-gin configs/model_reformer.gin.
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CHECKPOINT=""
OUTDIR=""
WINDOWS=""
PSMCFA=""

MODEL_TYPE="auto"
DEVICE="auto"
BATCH_SIZE=8
SAVE="probs"
REFORMER_GIN=""
LOSS_FUNCTION="CrossEntropyLoss"
N_CLASS=32

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --windows) WINDOWS="$2"; shift 2 ;;
    --psmcfa) PSMCFA="$2"; shift 2 ;;
    --model-type) MODEL_TYPE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --save) SAVE="$2"; shift 2 ;;
    --reformer-gin) REFORMER_GIN="$2"; shift 2 ;;
    --loss-function) LOSS_FUNCTION="$2"; shift 2 ;;
    --n-class) N_CLASS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done

[[ -n "$CHECKPOINT" ]] || die "--checkpoint is required"
[[ -n "$OUTDIR" ]] || die "--outdir is required"
if [[ -z "$WINDOWS" && -z "$PSMCFA" ]]; then
  die "Provide either --windows or --psmcfa"
fi
if [[ -n "$WINDOWS" && -n "$PSMCFA" ]]; then
  die "Provide only one of --windows or --psmcfa"
fi

ARGS=( --checkpoint "$CHECKPOINT" --outdir "$OUTDIR" --model-type "$MODEL_TYPE" --device "$DEVICE" --batch-size "$BATCH_SIZE" --save "$SAVE" --loss-function "$LOSS_FUNCTION" --n-class "$N_CLASS" )
if [[ -n "$WINDOWS" ]]; then
  ARGS+=( --windows "$WINDOWS" )
else
  ARGS+=( --psmcfa "$PSMCFA" )
fi
if [[ -n "$REFORMER_GIN" ]]; then
  ARGS+=( --reformer-gin "$REFORMER_GIN" )
fi

python "$REPO_ROOT/1000 геномов/infer_nn_from_psmcfa.py" "${ARGS[@]}"

