from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
)
from peft import TaskType

TEXT_TEXT_MODELS = {
    "t5": {
        "class": T5ForConditionalGeneration,
        "processor": AutoTokenizer,
        "8-bit": True,
        "target_modules": ["q", "v"],
        "task_type": TaskType.SEQ_2_SEQ_LM,
    },
}

TEXT_GEN_MODELS = {
    "opt": {
        "class": AutoModelForCausalLM,
        "processor": AutoTokenizer,
        "8-bit": False,
        "target_modules": ["k_proj", "v_proj", "q_proj", "out_proj"],
        "task_type": TaskType.CAUSAL_LM,
        "precision": 16,
    },
}
