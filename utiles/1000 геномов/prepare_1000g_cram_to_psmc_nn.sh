#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Prepare 1000 Genomes CRAM (GRCh37) for PSMC + deepgene NN format.

Pipeline:
  CRAM -> VCF (bcftools mpileup/call) -> consensus FASTQ (vcfutils.pl vcf2fq)
       -> .psmcfa (fastq_to_psmcfa.py) -> NN windows (.npz + manifest + meta)

Required tools in PATH:
  samtools bcftools vcfutils.pl python

Optional:
  psmc (or use ../psmc_checker/psmc)

Arguments:
  --sample  SAMPLE_ID
  --cram    /path/to/sample.cram
  --ref     /path/to/reference.fa   (must have .fai)
  --chrom   22|chr22|21,22|chr21,chr22  (single chromosome or comma-separated list)
  --outdir  /path/to/output_dir

Optional args:
  --bin-size 100
  --window-len 30000
  --step 30000
  --pad                  pad last NN window
  --min-mapq 30
  --min-baseq 20
  --minDP 10              (vcf2fq -d)
  --maxDP 100             (vcf2fq -D)
  --run-psmc              also run PSMC on produced .psmcfa
  --psmc-bin /path/to/psmc (default: ../psmc_checker/psmc if exists, else 'psmc' from PATH)
  --psmc-args "..."        (default matches psmc_checker/Makefile)
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

RUN_PSMC=0
PSMC_BIN=""
PSMC_ARGS="-N25 -t15 -r5 -p 4+25*2+4+6"

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

    --run-psmc) RUN_PSMC=1; shift 1 ;;
    --psmc-bin) PSMC_BIN="$2"; shift 2 ;;
    --psmc-args) PSMC_ARGS="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done

[[ -n "$SAMPLE" ]] || die "--sample is required"
[[ -n "$CRAM" ]] || die "--cram is required"
[[ -n "$REF" ]] || die "--ref is required"
[[ -n "$CHROM" ]] || die "--chrom is required"
[[ -n "$OUTDIR" ]] || die "--outdir is required"

is_url=0
if [[ "$CRAM" =~ ^https?:// || "$CRAM" =~ ^ftp:// ]]; then
  is_url=1
fi

if [[ "$is_url" == "0" ]]; then
  [[ -f "$CRAM" ]] || die "CRAM/BAM not found: $CRAM"
  [[ -f "$CRAM.crai" || -f "${CRAM%.cram}.crai" || -f "$CRAM.bai" || -f "${CRAM%.bam}.bam.bai" ]] || \
    echo "WARN: index (.crai/.bai) not found next to alignment; bcftools may fail if index is missing." >&2
else
  echo "INFO: Using remote alignment URL: $CRAM"
  echo "INFO: bcftools will try to fetch the index remotely (e.g. .crai/.bai) if available."
fi
[[ -f "$REF" ]] || die "Reference FASTA not found: $REF"
[[ -f "$REF.fai" ]] || die "Reference index not found: $REF.fai (run: samtools faidx $REF)"

need samtools
need bcftools
need vcfutils.pl
need python

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

  echo "[Chromosome $CHROM_ITEM] [1/4] Calling variants -> $VCF"
  # Notes:
  # - We use bcftools 'call -c' because it is compatible with the classic vcfutils.pl vcf2fq workflow.
  # - You may want to tune mpileup/call options for your data.
  bcftools mpileup \
    -Ou \
    -f "$REF" \
    -r "$CHROM_ITEM" \
    -q "$MIN_MAPQ" \
    -Q "$MIN_BASEQ" \
    "$CRAM" \
    | bcftools call -c -Ov -o "$VCF"

  echo "[Chromosome $CHROM_ITEM] [2/4] VCF -> consensus FASTQ -> $FQ"
  # vcf2fq uses depth filters:
  #  -d: minimum depth; -D: maximum depth; -Q: minimum RMS mapQ (historically)
  vcfutils.pl vcf2fq -d "$MIN_DP" -D "$MAX_DP" -Q "$MIN_BASEQ" "$VCF" > "$FQ"

  echo "[Chromosome $CHROM_ITEM] [3/4] FASTQ -> PSMCFA -> $PSMCFA"
  python "$REPO_ROOT/1000 геномов/fastq_to_psmcfa.py" \
    --fastq "$FQ" \
    --out "$PSMCFA" \
    --bin-size "$BIN_SIZE" \
    --q "$MIN_BASEQ" \
    --header "${SAMPLE}_${CHROM_ITEM}"

  PSMCFA_FILES+=("$PSMCFA")
  CHROM_NAMES+=("$CHROM_ITEM")
  
  if [[ "$RUN_PSMC" == "1" ]]; then
    if [[ -z "$PSMC_BIN" ]]; then
      if [[ -x "$REPO_ROOT/psmc_checker/psmc" ]]; then
        PSMC_BIN="$REPO_ROOT/psmc_checker/psmc"
      else
        PSMC_BIN="psmc"
      fi
    fi
    if [[ "$PSMC_BIN" != "psmc" ]]; then
      [[ -x "$PSMC_BIN" ]] || die "PSMC binary is not executable: $PSMC_BIN"
    else
      need psmc
    fi
    PSMC_OUT="$OUT_BASE/output.psmc"
    echo "[Chromosome $CHROM_ITEM] [PSMC] Running: $PSMC_BIN $PSMC_ARGS -o $PSMC_OUT $PSMCFA"
    "$PSMC_BIN" $PSMC_ARGS -o "$PSMC_OUT" "$PSMCFA"
  fi
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
  
  echo "[Combining] [4/4] PSMCFA -> NN windows -> $COMBINED_NPZ"
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
  
  echo "[4/4] PSMCFA -> NN windows -> $NN_NPZ"
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
  echo "  - $PSMCFA"
  echo "  - $NN_NPZ"
  echo "  - $NN_MANIFEST"
  echo "  - $NN_META"
fi

