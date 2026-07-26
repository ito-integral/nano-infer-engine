import torch
from transformers import AutoTokenizer

from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.generation.request import RequestStatus
from nano_infer_engine.generation.scheduler import ContinuousBatchingScheduler
from nano_infer_engine.loaders.llama import load_convert_hf_config, load_llama
from nano_infer_engine.models.llama import Llama3_2
from nano_infer_engine.paged_cache import PagedKVCache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

DEFAULT_MODEL_PATH = "/home/a/dm/models/Llama-3.2-1B-Instruct"
MAX_BATCH_SIZE = 2
BLOCK_SIZE = 16

nano_config = load_convert_hf_config(DEFAULT_MODEL_PATH)
nano_model = Llama3_2(nano_config).to(dtype=dtype)
load_llama(nano_model, DEFAULT_MODEL_PATH)
nano_model = nano_model.to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(
    DEFAULT_MODEL_PATH,
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
    ).input_ids.to(device)
    for formatted_prompt in formatted_prompts
)

generation_config = GenerationConfig(
    max_new_tokens=128,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
)

# Reserve enough physical blocks for the largest concurrently active requests.
request_block_counts = sorted(
    (
        input_ids.shape[1]
        + generation_config.max_new_tokens
        - 1
        + BLOCK_SIZE
        - 1
    )
    // BLOCK_SIZE
    for input_ids in prompt_input_ids
)
num_blocks = sum(request_block_counts[-MAX_BATCH_SIZE:])
paged_cache = PagedKVCache(
    num_blocks=num_blocks,
    block_size=BLOCK_SIZE,
    num_layers=len(nano_model.decoders),
    kv_head_num=nano_model.config.kv_head_num,
    head_dim=nano_model.config.head_dim,
    dtype=nano_model.embed.weight.dtype,
    device=device,
)
scheduler = ContinuousBatchingScheduler(
    nano_model,
    generation_config,
    paged_cache,
    max_batch_size=MAX_BATCH_SIZE,
)

requests = [
    scheduler.add_request(f"request-{index}", input_ids)
    for index, input_ids in enumerate(prompt_input_ids)
]
print(
    f"submitted={len(requests)}, active={scheduler.active_count}, "
    f"pending={scheduler.pending_count}"
)

step = 0
while scheduler.has_work:
    step += 1
    step_output = scheduler.step()
    if step == 1:
        print(
            f"after admission: active={scheduler.active_count}, "
            f"pending={scheduler.pending_count}"
        )
    for event in step_output.token_events:
        print(
            f"step={step}, sequence_id={event.sequence_id}, "
            f"token_id={event.token_id}"
        )
    for request in step_output.terminal_requests:
        print(
            f"step={step}, sequence_id={request.sequence_id}, "
            f"status={request.status.value}"
        )

for batch_index, (prompt, request) in enumerate(zip(prompts, requests)):
    print(f"\n=== batch {batch_index} ===")
    print(f"prompt: {prompt}")
    print(f"status: {request.status.value}")

    if request.status is RequestStatus.FAILED:
        print(f"error: {request.error}")
        continue

    prompt_length = request.prompt.shape[1]
    generated_ids = request.sequence[
        0,
        prompt_length : prompt_length + request.generated_tokens,
    ]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"generated_tokens: {request.generated_tokens}")
    print(f"stopped_by_eos: {request.finished}")
    print(f"output: {text}")
