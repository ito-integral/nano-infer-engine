import torch
from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.generation.greedy import greedy_generate
from transformers import AutoTokenizer

from nano_infer_engine.loaders.llama import load_convert_hf_config, load_llama
from nano_infer_engine.models.llama import Llama3_2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

DEFAULT_MODEL_PATH = "/home/a/dm/models/Llama-3.2-1B-Instruct"
nano_config = load_convert_hf_config(DEFAULT_MODEL_PATH)
nano_model = Llama3_2(nano_config).to(dtype=dtype)
load_llama(nano_model, DEFAULT_MODEL_PATH)
nano_model = nano_model.to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(
    DEFAULT_MODEL_PATH,
    local_files_only=True,
)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

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
model_inputs = tokenizer(
    formatted_prompts,
    add_special_tokens=False,
    padding=True,
    return_tensors="pt",
).to(device)
input_ids = model_inputs.input_ids
attention_mask = model_inputs.attention_mask

output = greedy_generate(
    nano_model,
    input_ids,
    GenerationConfig(
        max_new_tokens=128,
        eos_token_id=tokenizer.eos_token_id,
    ),
    attention_mask=attention_mask,
)

prompt_width = input_ids.shape[1]
for batch_index, prompt in enumerate(prompts):
    generated_count = output.generated_tokens[batch_index].item()
    generated_ids = output.sequences[
        batch_index,
        prompt_width : prompt_width + generated_count,
    ]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\n=== batch {batch_index} ===")
    print(f"prompt: {prompt}")
    print(f"generated_tokens: {generated_count}")
    print(f"stopped_by_eos: {output.stopped_by_eos[batch_index].item()}")
    print(f"output: {text}")
