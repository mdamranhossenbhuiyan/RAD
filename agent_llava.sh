export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - <<'PY'
import importlib, sys
for p in ["sentencepiece"]:
    try:
        importlib.import_module(p)
    except Exception:
        print(f"Missing {p}. Install with: pip install -U {p}")
        sys.exit(1)
print("Deps OK.")
PY

PYTHON=python
SCRIPT="llava_agentic_feedback_shard.py"

JSON_PATH="~/Synthetic_pids/HAM_1K.json"
TRAIN_DIR="~/Synthetic_pids/HAM_1K"
OUT_DIR="output_llava_vicuna_reid_agentic"

MODEL_ID="llava-hf/llava-v1.6-vicuna-7b-hf"
MODEL_DTYPE="float16"

USE_REF_STYLE=0

START_IDX=0
END_IDX=100000
PART_NAME="0K-100K"

BATCH_SIZE=2
MAX_IMAGE_SIDE=672

STAGE1_SAMPLES=14
KEEP_UNIQUE=12
MIN_VALID=10

TEMP=0.90
TOP_P=0.92
TOP_K=50
REP_PEN=1.20
NO_REPEAT=4

STAGE1_MAXTOK=40
CRITIC_MAXTOK=96
REFINE_MAXTOK=40
REFINE_SAMPLES=1

USE_CLIP_RERANK=0
CLIP_MODEL="ViT-L-14"
CLIP_PRETRAINED="openai"
CLIP_MODE="hybrid"
CLIP_CACHE=512
CLIP_ALPHA=0.35
CLIP_LAM=0.08
CLIP_MAX_WORDS=28

mkdir -p "$OUT_DIR"

REF_STYLE_FLAG=""
if [ "$USE_REF_STYLE" -eq 1 ]; then
  REF_STYLE_FLAG="--use_ref_style"
fi

CLIP_FLAG=""
if [ "$USE_CLIP_RERANK" -eq 1 ]; then
  CLIP_FLAG="--clip_rerank \
    --clip_model $CLIP_MODEL \
    --clip_pretrained $CLIP_PRETRAINED \
    --clip_mode $CLIP_MODE \
    --clip_cache_size $CLIP_CACHE \
    --clip_alpha $CLIP_ALPHA \
    --clip_penalty_lambda $CLIP_LAM \
    --clip_max_words $CLIP_MAX_WORDS"
fi

echo "======================================"
echo "MODEL_ID=$MODEL_ID"
echo "MODEL_DTYPE=$MODEL_DTYPE"
echo "TRAIN_DIR=$TRAIN_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "RANGE=$START_IDX -> $END_IDX"
echo "USE_REF_STYLE=$USE_REF_STYLE"
echo "MIN_VALID=$MIN_VALID"
echo "======================================"

CUDA_VISIBLE_DEVICES=5 $PYTHON "$SCRIPT" \
  --device cuda:0 \
  --train_dir "$TRAIN_DIR" \
  --out_dir "$OUT_DIR" \
  --json_path "$JSON_PATH" \
  --model_id "$MODEL_ID" \
  --model_dtype "$MODEL_DTYPE" \
  --part_name "$PART_NAME" \
  --start_idx "$START_IDX" \
  --end_idx "$END_IDX" \
  --batch_size "$BATCH_SIZE" \
  --max_image_side "$MAX_IMAGE_SIDE" \
  $REF_STYLE_FLAG \
  --stage1_samples "$STAGE1_SAMPLES" \
  --stage1_max_new_tokens "$STAGE1_MAXTOK" \
  --critic_max_new_tokens "$CRITIC_MAXTOK" \
  --refine_samples "$REFINE_SAMPLES" \
  --refine_max_new_tokens "$REFINE_MAXTOK" \
  --keep_unique_candidates "$KEEP_UNIQUE" \
  --min_valid_captions "$MIN_VALID" \
  --temperature "$TEMP" \
  --top_p "$TOP_P" \
  --top_k "$TOP_K" \
  --repetition_penalty "$REP_PEN" \
  --no_repeat_ngram_size "$NO_REPEAT" \
  $CLIP_FLAG \
  --resume \
  --save_every_batch
