# nano-infer-engine

## Install

```bash
uv python install 3.13
uv sync
```

## Start the single-GPU service

```bash
CUDA_VISIBLE_DEVICES=0 \
NANO_MODEL_PATH=/home/a/dm/models/Llama-3.2-1B-Instruct \
NANO_SERVED_MODEL_NAME=llama3.2-1b \
NANO_MAX_MODEL_LEN=29600 \
NANO_GPU_MEMORY_UTILIZATION=0.9 \
NANO_DEVICE=cuda:0 \
NANO_MAX_BATCH_SIZE=16 \
NANO_MAX_NEW_TOKENS=256 \
NANO_PREFILL_CHUNK_SIZE=128 \
NANO_MAX_PREFILL_TOKENS_PER_STEP=512 \
uv run uvicorn nano_infer_engine.service.server:app \
  --host 0.0.0.0 \
  --port 8000
```

## Start the two-GPU P/D service

```bash
NANO_MODEL_PATH=/home/a/dm/models/Llama-3.2-1B-Instruct \
NANO_SERVED_MODEL_NAME=llama3.2-1b \
NANO_MAX_MODEL_LEN=29600 \
NANO_PREFILL_DEVICE=cuda:0 \
NANO_DECODE_DEVICE=cuda:1 \
NANO_PREFILL_GPU_MEMORY_UTILIZATION=0.9 \
NANO_DECODE_GPU_MEMORY_UTILIZATION=0.9 \
NANO_MAX_BATCH_SIZE=16 \
NANO_MAX_NEW_TOKENS=256 \
uv run uvicorn nano_infer_engine.service.pd_server:app \
  --host 0.0.0.0 \
  --port 8000
```
