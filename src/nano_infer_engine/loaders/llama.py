import json
from pathlib import Path
from typing import Any
from safetensors import safe_open
import torch
from nano_infer_engine.models.llama import Llama3_2, LlamaConfig, RopeScalingConfig


_REQUIRED_CONFIG_KEYS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "rms_norm_eps",
    "max_position_embeddings",
    "bos_token_id",
    "eos_token_id",
)


def _convert_rope_scaling(
    hf_config: dict[str, Any],
) -> tuple[str, RopeScalingConfig]:
    hf_rope_scaling = hf_config.get("rope_scaling")
    if hf_rope_scaling is None:
        return "default", RopeScalingConfig()

    if not isinstance(hf_rope_scaling, dict):
        raise TypeError("'rope_scaling' must be an object or null")

    rope_type = hf_rope_scaling.get("rope_type", hf_rope_scaling.get("type", "default"))
    if rope_type != "llama3":
        raise ValueError(f"Unsupported Hugging Face RoPE type: {rope_type!r}")

    required_keys = (
        "factor",
        "low_freq_factor",
        "high_freq_factor",
        "original_max_position_embeddings",
    )
    missing_keys = [key for key in required_keys if key not in hf_rope_scaling]
    if missing_keys:
        raise KeyError(
            "Missing required Llama 3 RoPE config fields: " + ", ".join(missing_keys)
        )

    return rope_type, RopeScalingConfig(
        factor=float(hf_rope_scaling["factor"]),
        low_freq_factor=float(hf_rope_scaling["low_freq_factor"]),
        high_freq_factor=float(hf_rope_scaling["high_freq_factor"]),
        original_max_position_embeddings=int(
            hf_rope_scaling["original_max_position_embeddings"]
        ),
    )


def load_convert_hf_config(config_path: str | Path) -> LlamaConfig:
    config_path = Path(config_path)
    if config_path.is_dir():
        config_path = config_path / "config.json"

    with config_path.open("r", encoding="utf-8") as file:
        hf_config = json.load(file)

    if not isinstance(hf_config, dict):
        raise TypeError(f"Hugging Face config must be a JSON object: {config_path}")

    model_type = hf_config.get("model_type")
    if model_type != "llama":
        raise ValueError(
            f"Expected a Llama config with model_type='llama', got {model_type!r}"
        )

    missing_keys = [key for key in _REQUIRED_CONFIG_KEYS if key not in hf_config]
    if missing_keys:
        raise KeyError(
            "Missing required Hugging Face Llama config fields: "
            + ", ".join(missing_keys)
        )

    hidden_size = int(hf_config["hidden_size"])
    q_head_num = int(hf_config["num_attention_heads"])
    derived_head_dim = hidden_size // q_head_num
    hf_head_dim = int(hf_config.get("head_dim", derived_head_dim))
    if hf_head_dim != derived_head_dim:
        raise ValueError(
            "This engine derives head_dim as hidden_size / num_attention_heads, "
            f"but the Hugging Face config specifies head_dim={hf_head_dim} "
            f"instead of {derived_head_dim}"
        )

    rope_type, rope_scaling = _convert_rope_scaling(hf_config)

    return LlamaConfig(
        vocab_size=int(hf_config["vocab_size"]),
        hidden_size=hidden_size,
        mlp_inner_size=int(hf_config["intermediate_size"]),
        num_layers=int(hf_config["num_hidden_layers"]),
        q_head_num=q_head_num,
        kv_head_num=int(hf_config["num_key_value_heads"]),
        norm_eps=float(hf_config["rms_norm_eps"]),
        rope_type=rope_type,
        rope_theta=float(hf_config.get("rope_theta", 10000.0)),
        max_seq_len=int(hf_config["max_position_embeddings"]),
        rope_scaling=rope_scaling,
        hidden_act=hf_config.get("hidden_act", "silu"),
        attention_bias=bool(hf_config.get("attention_bias", False)),
        mlp_bias=bool(hf_config.get("mlp_bias", False)),
        attention_dropout=float(hf_config.get("attention_dropout", 0.0)),
        tie_word_embeddings=bool(hf_config.get("tie_word_embeddings", False)),
        initializer_range=float(hf_config.get("initializer_range", 0.02)),
        bos_token_id=int(hf_config["bos_token_id"]),
        eos_token_id=int(hf_config["eos_token_id"]),
    )


def map_llama_weight_name(hf_name: str) -> str:
    global_weight_names = {
        "model.embed_tokens.weight": "embed.weight",
        "model.norm.weight": "final_rms.weight",
        "lm_head.weight": "lm_head.weight",
    }
    if hf_name in global_weight_names:
        return global_weight_names[hf_name]

    layer_prefix = "model.layers."
    if not hf_name.startswith(layer_prefix):
        raise KeyError(f"Unsupported Hugging Face Llama weight: {hf_name}")

    layer_and_suffix = hf_name.removeprefix(layer_prefix)
    layer_index, separator, suffix = layer_and_suffix.partition(".")
    if not separator or not layer_index.isdigit():
        raise KeyError(f"Invalid Hugging Face Llama layer weight: {hf_name}")

    layer_weight_names = {
        "input_layernorm.weight": "pre_norm.weight",
        "post_attention_layernorm.weight": "post_norm.weight",
        "self_attn.q_proj.weight": "attn.q_proj.weight",
        "self_attn.k_proj.weight": "attn.k_proj.weight",
        "self_attn.v_proj.weight": "attn.v_proj.weight",
        "self_attn.o_proj.weight": "attn.o_proj.weight",
        "mlp.gate_proj.weight": "ffn.gate_proj.weight",
        "mlp.up_proj.weight": "ffn.up_proj.weight",
        "mlp.down_proj.weight": "ffn.down_proj.weight",
    }
    try:
        mapped_suffix = layer_weight_names[suffix]
    except KeyError:
        raise KeyError(f"Unsupported Hugging Face Llama weight: {hf_name}") from None

    return f"decoders.{layer_index}.{mapped_suffix}"


def load_llama(model, shard_path):
    model_state = model.state_dict()

    if isinstance(shard_path, str):
        shard_path = Path(shard_path)

    with torch.no_grad():
        for tensor_file in sorted(shard_path.glob("*.safetensors")):
            with safe_open(tensor_file, framework="pt", device="cpu") as file:
                for hf_name in file.keys():
                    mapped_name = map_llama_weight_name(hf_name)
                    if mapped_name not in model_state:
                        raise KeyError(
                            f"Mapped weight name {mapped_name!r} not found in model state"
                        )
                    tensor = file.get_tensor(hf_name)
                    target = model_state[mapped_name]

                    if tensor.shape != target.shape:
                        raise ValueError("复制失败，形状不一样")

                    target.copy_(tensor)

    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.embed.weight


if __name__ == "__main__":
    hf_config_path = "/home/a/dm/models/Llama-3.2-1B-Instruct/config.json"
    with open(hf_config_path, "r", encoding="utf-8") as file:
        hf_config = json.load(file)
    print(hf_config)
    my_config = load_convert_hf_config(hf_config_path)

    model = Llama3_2(my_config)
    load_llama(model, r"/home/a/dm/models/Llama-3.2-1B-Instruct")
    model_state = model.state_dict()
