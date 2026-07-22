import torch
from nano_infer_engine.generation.config import GenerationConfig
from nano_infer_engine.generation.greedy import greedy_generate
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_infer_engine.loaders.llama import load_convert_hf_config, load_llama
from nano_infer_engine.models.llama import Llama3_2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MODEL_PATH = "/home/a/dm/models/Llama-3.2-1B-Instruct"
nano_config = load_convert_hf_config(DEFAULT_MODEL_PATH)
nano_model = Llama3_2(nano_config).to(dtype=torch.bfloat16)
load_llama(nano_model, DEFAULT_MODEL_PATH)
nano_model = nano_model.to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(
    DEFAULT_MODEL_PATH,
    local_files_only=True,
)

messages = [{"role": "user", "content": "介绍下你自己"}]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(device)

output = greedy_generate(
    nano_model,
    input_ids,
    GenerationConfig(
        max_new_tokens=2000,
        eos_token_id=tokenizer.eos_token_id,
    ),
)

generated_ids = output.sequences[:, input_ids.shape[1] :]
text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(text)
