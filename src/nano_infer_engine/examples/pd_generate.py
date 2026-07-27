import torch
from transformers import AutoTokenizer

from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.generation.pd_greedy import pd_greedy_generate
from nano_infer_engine.loaders.llama import load_convert_hf_config, load_llama
from nano_infer_engine.models.llama import Llama3_2
from nano_infer_engine.paged_cache import PagedKVCache


MODEL_PATH = "/home/a/dm/models/Llama-3.2-1B-Instruct"
BLOCK_SIZE = 16
MAX_NEW_TOKENS = 128

if torch.cuda.device_count() < 2:
    raise RuntimeError("P/D generation requires at least two CUDA devices")

prefill_device = torch.device("cuda:0")
decode_device = torch.device("cuda:1")
dtype = torch.bfloat16
model_config = load_convert_hf_config(MODEL_PATH)

prefill_model = Llama3_2(model_config).to(dtype=dtype)
load_llama(prefill_model, MODEL_PATH)
prefill_model = prefill_model.to(prefill_device).eval()

decode_model = Llama3_2(model_config).to(dtype=dtype)
load_llama(decode_model, MODEL_PATH)
decode_model = decode_model.to(decode_device).eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)
prompt = "介绍下你自己"
formatted_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
input_ids = tokenizer(
    formatted_prompt,
    add_special_tokens=False,
    return_tensors="pt",
).input_ids.to(prefill_device)

generation_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
)
num_blocks = (
    input_ids.shape[1] + MAX_NEW_TOKENS - 1 + BLOCK_SIZE - 1
) // BLOCK_SIZE


def build_cache(device: torch.device) -> PagedKVCache:
    return PagedKVCache(
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_layers=len(prefill_model.decoders),
        kv_head_num=model_config.kv_head_num,
        head_dim=model_config.head_dim,
        dtype=dtype,
        device=device,
    )


prefill_cache = build_cache(prefill_device)
decode_cache = build_cache(decode_device)
output = pd_greedy_generate(
    prefill_model,
    decode_model,
    input_ids,
    generation_config,
    prefill_cache,
    decode_cache,
    sequence_id="request-a",
)

generated_ids = output.sequences[0, input_ids.shape[1] :]
print(f"prompt: {prompt}")
print(f"generated_tokens: {int(output.generated_tokens[0])}")
print(f"stopped_by_eos: {bool(output.stopped_by_eos[0])}")
print(tokenizer.decode(generated_ids, skip_special_tokens=True))
