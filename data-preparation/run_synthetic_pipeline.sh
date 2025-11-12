#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_synthetic_pipeline.sh --input <path> [options]

Required:
  -i, --input <path>        File or directory to ingest.

Optional:
      --content-type <val>  Content type for generation (qa|summary|cot|cot-enhance|multimodal-qa).
      --num-pairs <int>     Target number of examples to generate (create stage).
      --threshold <float>   Quality threshold for curation.
  -v, --verbose             Forward verbose flag to CLI commands.
  -h, --help                Show this help text.
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

INPUT_PATH="./data/input"
CONTENT_TYPE="qa"
NUM_PAIRS=""
THRESHOLD="8.5"
VERBOSE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      shift
      INPUT_PATH="${1:-}"
      ;;
    --content-type|--type)
      shift
      CONTENT_TYPE="${1:-}"
      ;;
    --num-pairs)
      shift
      NUM_PAIRS="${1:-}"
      ;;
    --threshold)
      shift
      THRESHOLD="${1:-}"
      ;;
    -v|--verbose)
      VERBOSE=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ -z "$INPUT_PATH" ]]; then
  echo "Error: --input is required." >&2
  usage
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

PARSED_DIR="data/parsed"
GENERATED_DIR="data/generated"
CURATED_DIR="data/curated"
FINAL_DIR="data/final"

timestamp_ist() {
  TZ='Asia/Kolkata' date '+%Y-%m-%d %H:%M:%S %Z'
}

run_cli() {
  local args=("$@")
  if [[ "$VERBOSE" == true ]]; then
    args+=(--verbose)
  fi
  local cmd=("synthetic-data-kit" "--config" "configs/config.yaml")
  cmd+=("${args[@]}")
  printf '###############\n'
  printf '%s CMD: %s\n' "$(timestamp_ist)" "$(printf '%q ' "${cmd[@]}")"
  printf '###############\n'
  "${cmd[@]}"
}

CREATE_ARGS=("create" "$PARSED_DIR" "--type" "$CONTENT_TYPE")
if [[ -n "$NUM_PAIRS" ]]; then
  CREATE_ARGS+=(--num-pairs "$NUM_PAIRS")
fi

CURATE_ARGS=("curate" "$GENERATED_DIR")
CURATE_ARGS+=(--threshold "$THRESHOLD")

echo "Ingesting '$INPUT_PATH'..."
run_cli ingest "$INPUT_PATH"

echo "Generating synthetic data..."
run_cli "${CREATE_ARGS[@]}"

echo "Curating generated data..."
run_cli "${CURATE_ARGS[@]}"

echo "Converting curated data to JSONL..."
run_cli save-as "$CURATED_DIR" --format jsonl

# Post-process JSONL files to instruction/context/response schema
python - <<'PY' "$CURATED_DIR" "$GENERATED_DIR" "$FINAL_DIR"
import json
import sys
from pathlib import Path

curated_dir = Path(sys.argv[1])
generated_dir = Path(sys.argv[2])
final_dir = Path(sys.argv[3])

for curated_path in sorted(curated_dir.glob("*.json")):
    base_name = curated_path.stem  # e.g., foo_qa_pairs_cleaned
    final_path = final_dir / f"{base_name}.jsonl"
    if not final_path.exists():
        continue

    try:
        curated_data = json.loads(curated_path.read_text(encoding="utf-8"))
    except Exception:
        continue

    pairs = curated_data.get("qa_pairs") or curated_data.get("filtered_pairs") or []
    if not pairs:
        continue

    # Build lookup from generated file for missing context
    generated_lookup = {}
    generated_path = generated_dir / f"{base_name.replace('_cleaned', '')}.json"
    if generated_path.exists():
        try:
            generated_data = json.loads(generated_path.read_text(encoding="utf-8"))
            for item in generated_data.get("qa_pairs", []):
                key = item.get("instruction") or item.get("question") or ""
                if key:
                    generated_lookup[key] = item.get("context") or ""
        except Exception:
            pass

    transformed = []
    for pair in pairs:
        instruction = pair.get("instruction") or pair.get("question") or ""
        response = pair.get("response") or pair.get("answer") or ""
        context = pair.get("context") or generated_lookup.get(instruction, "")

        if not instruction or not response:
            continue

        transformed.append({
            "instruction": instruction,
            "context": context,
            "response": response
        })

    if not transformed:
        continue

    with final_path.open("w", encoding="utf-8") as out_f:
        for item in transformed:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
PY

echo "Pipeline complete. JSONL files under '$FINAL_DIR':"
find "$FINAL_DIR" -type f -name '*.jsonl' -print 2>/dev/null || echo "  (no JSONL files found)"
