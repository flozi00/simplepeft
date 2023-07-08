from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LlamaTokenizer,
)
from peft import TaskType

TEXT_TEXT_MODELS = {
    "t5": {
        "class": AutoModelForSeq2SeqLM,
        "processor": AutoTokenizer,
        "8-bit": True,
        "target_modules": ["q", "v"],
        "task_type": TaskType.SEQ_2_SEQ_LM,
    },
    "longt5": {
        "class": AutoModelForSeq2SeqLM,
        "processor": AutoTokenizer,
        "8-bit": True,
        "target_modules": ["q", "v"],
        "task_type": TaskType.SEQ_2_SEQ_LM,
    },
    "bart": {
        "class": AutoModelForSeq2SeqLM,
        "processor": AutoTokenizer,
        "8-bit": False,
        "target_modules": [
            "v_proj",
            "q_proj",
        ],
        "task_type": TaskType.SEQ_2_SEQ_LM,
    },
    "led": {
        "class": AutoModelForSeq2SeqLM,
        "processor": AutoTokenizer,
        "8-bit": True,
        "target_modules": [
            "query",
            "key",
            "value",
            "query_global",
            "key_global",
            "value_global",
        ],
        "task_type": TaskType.SEQ_2_SEQ_LM,
    },
}

TEXT_GEN_MODELS = {
    "opt": {
        "class": AutoModelForCausalLM,
        "processor": AutoTokenizer,
        "8-bit": True,
        "target_modules": ["v_proj", "q_proj"],
        "task_type": TaskType.CAUSAL_LM,
    },
    "rwkv": {
        "class": AutoModelForCausalLM,
        "processor": AutoTokenizer,
        "8-bit": True,
        "target_modules": [
            "key",
            "value",
        ],
        "task_type": TaskType.CAUSAL_LM,
    },
    "gpt_neox": {
        "class": AutoModelForCausalLM,
        "processor": AutoTokenizer,
        "8-bit": True,
        "target_modules": ["query_key_value"],
        "task_type": TaskType.CAUSAL_LM,
    },
    "RefinedWebModel": {
        "class": AutoModelForCausalLM,
        "processor": AutoTokenizer,
        "8-bit": True,
        "target_modules": [
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
            "lm_head",
            "word_embeddings",
        ],
        "task_type": TaskType.CAUSAL_LM,
    },
    "llama": {
        "class": AutoModelForCausalLM,
        "processor": LlamaTokenizer,
        "8-bit": True,
        "target_modules": [
            "gate_proj",
            "down_proj",
            "up_proj",
            "v_proj",
            "q_proj",
            "k_proj",
            "o_proj",
            "embed_tokens",
        ],
        "task_type": TaskType.CAUSAL_LM,
    },
}
