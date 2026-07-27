import torch
from transformers import AutoTokenizer

from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.generation.pd_scheduler import (
    PDContinuousBatchingScheduler,
)
from nano_infer_engine.generation.request import RequestStatus
from nano_infer_engine.loaders.llama import load_convert_hf_config, load_llama
from nano_infer_engine.models.llama import Llama3_2
from nano_infer_engine.paged_cache import PagedKVCache


MODEL_PATH = "/home/a/dm/models/Llama-3.2-1B-Instruct"
MAX_BATCH_SIZE = 2
MAX_NEW_TOKENS = 128
BLOCK_SIZE = 16

if torch.cuda.device_count() < 2:
    raise RuntimeError("P/D batching requires at least two CUDA devices")

prefill_device = torch.device("cuda:0")
decode_device = torch.device("cuda:1")
dtype = torch.bfloat16
model_config = load_convert_hf_config(MODEL_PATH)


def load_model(device: torch.device) -> Llama3_2:
    model = Llama3_2(model_config).to(dtype=dtype)
    load_llama(model, MODEL_PATH)
    return model.to(device).eval()


prefill_model = load_model(prefill_device)
decode_model = load_model(decode_device)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

prompts = [
    "介绍下你自己",
    "请用一句话介绍下你自己",
    "如何快速减肥？请给出三条健康且可持续的建议。",
]
formatted_prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for prompt in prompts
]
prompt_input_ids = tuple(
    tokenizer(
        formatted_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(prefill_device)
    for formatted_prompt in formatted_prompts
)
generation_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
)

prefill_num_blocks = max(
    (input_ids.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
    for input_ids in prompt_input_ids
)
decode_request_blocks = sorted(
    (
        input_ids.shape[1]
        + MAX_NEW_TOKENS
        - 1
        + BLOCK_SIZE
        - 1
    )
    // BLOCK_SIZE
    for input_ids in prompt_input_ids
)
decode_num_blocks = sum(decode_request_blocks[-MAX_BATCH_SIZE:])


def build_cache(num_blocks: int, device: torch.device) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_layers=len(prefill_model.decoders),
        kv_head_num=model_config.kv_head_num,
        head_dim=model_config.head_dim,
        dtype=dtype,
        device=device,
    )


scheduler = PDContinuousBatchingScheduler(
    prefill_model,
    decode_model,
    generation_config,
    build_cache(prefill_num_blocks, prefill_device),
    build_cache(decode_num_blocks, decode_device),
    max_batch_size=MAX_BATCH_SIZE,
)
requests = [
    scheduler.add_request(f"request-{index}", input_ids)
    for index, input_ids in enumerate(prompt_input_ids)
]

step = 0
while scheduler.has_work:
    step += 1
    output = scheduler.step()
    print(
        f"step={step}, active={scheduler.active_count}, "
        f"pending={scheduler.pending_count}"
    )
    for event in output.token_events:
        print(f"  token: {event.sequence_id} -> {event.token_id}")
    for request in output.terminal_requests:
        print(f"  terminal: {request.sequence_id} -> {request.status.value}")

for prompt, request in zip(prompts, requests):
    print(f"\n=== {request.sequence_id} ===")
    print(f"prompt: {prompt}")
    if request.status is RequestStatus.FAILED:
        print(f"error: {request.error}")
        continue
    generated_ids = request.sequence[0, request.prompt.shape[1] :]
    print(f"generated_tokens: {request.generated_tokens}")
    print(f"stopped_by_eos: {request.finished}")
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))
