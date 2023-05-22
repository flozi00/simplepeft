from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
)
from peft import TaskType

TEXT_TEXT_MODELS = {
    "t5": {
        "class": T5ForConditionalGeneration,
        "processor": AutoTokenizer,
        "8-bit": True,
        "target_modules": ["q", "v"],
        "task_type": TaskType.SEQ_2_SEQ_LM,
        "gradient_accumulation": 1,
    },
    "bart": { 
        "class": AutoModelForSeq2SeqLM, 
            "processor": AutoTokenizer, 
            "8-bit": False, 
            "target_modules": ["v_proj", "q_proj",], 
            "task_type": TaskType.SEQ_2_SEQ_LM, 
            "gradient_accumulation": 4, 
        },
}

TEXT_GEN_MODELS = {
    "opt": {
        "class": AutoModelForCausalLM,
        "processor": AutoTokenizer,
        "8-bit": True,
        "target_modules": ["v_proj", "q_proj"],
        "task_type": TaskType.CAUSAL_LM,
        "gradient_accumulation": 4,
    },
    "rwkv": {
        "class": AutoModelForCausalLM,
        "processor": AutoTokenizer,
        "8-bit": False,
        "target_modules": ["key", "value",],
        "task_type": TaskType.CAUSAL_LM,
        "gradient_accumulation": 1,
    },
    "gpt_neox": {
        "class": AutoModelForCausalLM,
        "processor": AutoTokenizer,
        "8-bit": True,
        "target_modules": ["query_key_value"],
        "task_type": TaskType.CAUSAL_LM,
        "gradient_accumulation": 4,
    },
}
