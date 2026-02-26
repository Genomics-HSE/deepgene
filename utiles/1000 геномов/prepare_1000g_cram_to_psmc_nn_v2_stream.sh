#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
V2 (streaming): Prepare 1000 Genomes CRAM/BAM (GRCh37/hs37d5) for PSMC + deepgene NN format
WITHOUT writing a giant calls.vcf to disk.

Pipeline:
  CRAM/BAM -> VCF stream (bcftools mpileup/call) -> consensus FASTQ (vcfutils.pl vcf2fq via FIFO)
           -> .psmcfa (fastq_to_psmcfa.py) -> NN windows (.npz + manifest + meta)

Required tools in PATH:
  samtools bcftools vcfutils.pl python

Args:
  --sample  SAMPLE_ID
  --cram    /path/to/sample.{cram|bam}  OR ftp/http URL
  --ref     /path/to/hs37d5.fa   (must have .fai)
  --chrom   22|chr22|21,22|chr21,chr22  (single chromosome or comma-separated list)
  --outdir  /path/to/output_dir

Optional:
  --bin-size 100
  --window-len 30000
  --step 30000
  --pad                  pad last NN window
  --min-mapq 30
  --min-baseq 20
  --minDP 10              (vcf2fq -d)
  --maxDP 100             (vcf2fq -D)
  --keep-vcf              also save calls.vcf (debug; off by default)
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

SAMPLE=""
CRAM=""
REF=""
CHROM=""
OUTDIR=""

BIN_SIZE=100
WINDOW_LEN=30000
STEP=30000
PAD=0

MIN_MAPQ=30
MIN_BASEQ=20

MIN_DP=10
MAX_DP=100

KEEP_VCF=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample) SAMPLE="$2"; shift 2 ;;
    --cram) CRAM="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --chrom) CHROM="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;

    --bin-size) BIN_SIZE="$2"; shift 2 ;;
    --window-len) WINDOW_LEN="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --pad) PAD=1; shift 1 ;;

    --min-mapq) MIN_MAPQ="$2"; shift 2 ;;
    --min-baseq) MIN_BASEQ="$2"; shift 2 ;;
    --minDP) MIN_DP="$2"; shift 2 ;;
    --maxDP) MAX_DP="$2"; shift 2 ;;

    --keep-vcf) KEEP_VCF=1; shift 1 ;;

    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done

[[ -n "$SAMPLE" ]] || die "--sample is required"
[[ -n "$CRAM" ]] || die "--cram is required"
[[ -n "$REF" ]] || die "--ref is required"
[[ -n "$CHROM" ]] || die "--chrom is required"
[[ -n "$OUTDIR" ]] || die "--outdir is required"

need samtools
need bcftools
need vcfutils.pl
need python

[[ -f "$REF" ]] || die "Reference FASTA not found: $REF"
[[ -f "$REF.fai" ]] || die "Reference index not found: $REF.fai (run: samtools faidx $REF)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse chromosomes (comma-separated or single)
IFS=',' read -ra CHROM_ARRAY <<< "$CHROM"
N_CHROMS=${#CHROM_ARRAY[@]}

# Process each chromosome
PSMCFA_FILES=()
CHROM_NAMES=()

for CHROM_ITEM in "${CHROM_ARRAY[@]}"; do
  CHROM_ITEM="${CHROM_ITEM#"${CHROM_ITEM%%[![:space:]]*}"}"  # trim leading whitespace
  CHROM_ITEM="${CHROM_ITEM%"${CHROM_ITEM##*[![:space:]]}"}"  # trim trailing whitespace
  
  OUT_BASE="$OUTDIR/$SAMPLE/chr${CHROM_ITEM}"
  if [[ "$CHROM_ITEM" == chr* ]]; then
    OUT_BASE="$OUTDIR/$SAMPLE/${CHROM_ITEM}"
  fi
  mkdir -p "$OUT_BASE"

  VCF="$OUT_BASE/calls.vcf"
  FQ="$OUT_BASE/consensus.fq"
  PSMCFA="$OUT_BASE/input.psmcfa"

  echo "[Chromosome $CHROM_ITEM] [1/3] Calling variants (stream) + consensus FASTQ -> $FQ"

  FIFO="$OUT_BASE/.calls.vcf.fifo"
  cleanup_chrom() {
    rm -f "$FIFO" 2>/dev/null || true
  }
  trap cleanup_chrom EXIT
  rm -f "$FIFO"
  mkfifo "$FIFO"

  # vcf2fq consumes a VCF *file*; FIFO lets us stream VCF without saving it.
  vcfutils.pl vcf2fq -d "$MIN_DP" -D "$MAX_DP" -Q "$MIN_BASEQ" "$FIFO" > "$FQ" &
  VCF2FQ_PID=$!

  # Optionally tee to calls.vcf for debugging.
  if [[ "$KEEP_VCF" == "1" ]]; then
    ( bcftools mpileup -Ou -f "$REF" -r "$CHROM_ITEM" -q "$MIN_MAPQ" -Q "$MIN_BASEQ" "$CRAM" \
        | bcftools call -c -Ov \
        | tee "$VCF" > "$FIFO" ) || true
  else
    ( bcftools mpileup -Ou -f "$REF" -r "$CHROM_ITEM" -q "$MIN_MAPQ" -Q "$MIN_BASEQ" "$CRAM" \
        | bcftools call -c -Ov > "$FIFO" ) || true
  fi

  wait "$VCF2FQ_PID"

  if [[ ! -s "$FQ" ]]; then
    die "consensus FASTQ is empty for chromosome $CHROM_ITEM: $FQ"
  fi

  echo "[Chromosome $CHROM_ITEM] [2/3] FASTQ -> PSMCFA -> $PSMCFA"
  python "$REPO_ROOT/1000 геномов/fastq_to_psmcfa.py" \
    --fastq "$FQ" \
    --out "$PSMCFA" \
    --bin-size "$BIN_SIZE" \
    --q "$MIN_BASEQ" \
    --header "${SAMPLE}_${CHROM_ITEM}"

  PSMCFA_FILES+=("$PSMCFA")
  CHROM_NAMES+=("$CHROM_ITEM")
done

# Combine all chromosomes if multiple
if [[ $N_CHROMS -gt 1 ]]; then
  echo "[Combining] Merging ${N_CHROMS} chromosomes..."
  COMBINED_DIR="$OUTDIR/$SAMPLE/combined"
  mkdir -p "$COMBINED_DIR"
  
  COMBINED_NPZ="$COMBINED_DIR/nn_windows.npz"
  COMBINED_MANIFEST="$COMBINED_DIR/nn_windows_manifest.tsv"
  COMBINED_META="$COMBINED_DIR/nn_meta.json"
  
  CHROM_STR=$(IFS=','; echo "${CHROM_NAMES[*]}")
  
  PAD_FLAG=""
  if [[ "$PAD" == "1" ]]; then PAD_FLAG="--pad"; fi
  
  python "$REPO_ROOT/1000 геномов/psmcfa_to_nn_windows.py" \
    --combine-psmcfa "${PSMCFA_FILES[@]}" \
    --bin-size "$BIN_SIZE" \
    --chrom "$CHROM_STR" \
    --sample "$SAMPLE" \
    --window-len "$WINDOW_LEN" \
    --step "$STEP" \
    $PAD_FLAG \
    --out "$COMBINED_NPZ" \
    --manifest "$COMBINED_MANIFEST" \
    --meta "$COMBINED_META"
  
  echo "Done."
  echo "Combined outputs:"
  echo "  - $COMBINED_NPZ"
  echo "  - $COMBINED_MANIFEST"
  echo "  - $COMBINED_META"
  echo ""
  echo "Individual chromosome outputs are in:"
  for CHROM_ITEM in "${CHROM_ARRAY[@]}"; do
    CHROM_ITEM="${CHROM_ITEM#"${CHROM_ITEM%%[![:space:]]*}"}"
    CHROM_ITEM="${CHROM_ITEM%"${CHROM_ITEM##*[![:space:]]}"}"
    OUT_BASE="$OUTDIR/$SAMPLE/chr${CHROM_ITEM}"
    if [[ "$CHROM_ITEM" == chr* ]]; then
      OUT_BASE="$OUTDIR/$SAMPLE/${CHROM_ITEM}"
    fi
    echo "  - $OUT_BASE/"
  done
else
  # Single chromosome - use original output location
  CHROM_ITEM="${CHROM_ARRAY[0]}"
  CHROM_ITEM="${CHROM_ITEM#"${CHROM_ITEM%%[![:space:]]*}"}"
  CHROM_ITEM="${CHROM_ITEM%"${CHROM_ITEM##*[![:space:]]}"}"
  
  OUT_BASE="$OUTDIR/$SAMPLE/chr${CHROM_ITEM}"
  if [[ "$CHROM_ITEM" == chr* ]]; then
    OUT_BASE="$OUTDIR/$SAMPLE/${CHROM_ITEM}"
  fi
  
  NN_NPZ="$OUT_BASE/nn_windows.npz"
  NN_MANIFEST="$OUT_BASE/nn_windows_manifest.tsv"
  NN_META="$OUT_BASE/nn_meta.json"
  
  echo "[3/3] PSMCFA -> NN windows -> $NN_NPZ"
  PAD_FLAG=""
  if [[ "$PAD" == "1" ]]; then PAD_FLAG="--pad"; fi

  python "$REPO_ROOT/1000 геномов/psmcfa_to_nn_windows.py" \
    --psmcfa "${PSMCFA_FILES[0]}" \
    --bin-size "$BIN_SIZE" \
    --chrom "$CHROM_ITEM" \
    --sample "$SAMPLE" \
    --window-len "$WINDOW_LEN" \
    --step "$STEP" \
    $PAD_FLAG \
    --out "$NN_NPZ" \
    --manifest "$NN_MANIFEST" \
    --meta "$NN_META"

  echo "Done."
  echo "Outputs:"
  echo "  - $OUT_BASE/consensus.fq"
  echo "  - $OUT_BASE/input.psmcfa"
  echo "  - $NN_NPZ"
  if [[ "$KEEP_VCF" == "1" ]]; then
    echo "  - $OUT_BASE/calls.vcf"
  fi
fi

