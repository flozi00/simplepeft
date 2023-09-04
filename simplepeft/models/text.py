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
        "task_type": TaskType.SEQ_2_SEQ_LM,
    },
    "longt5": {
        "class": AutoModelForSeq2SeqLM,
        "processor": AutoTokenizer,
        "task_type": TaskType.SEQ_2_SEQ_LM,
    },
    "bart": {
        "class": AutoModelForSeq2SeqLM,
        "processor": AutoTokenizer,
        "task_type": TaskType.SEQ_2_SEQ_LM,
    },
    "led": {
        "class": AutoModelForSeq2SeqLM,
        "processor": AutoTokenizer,
        "task_type": TaskType.SEQ_2_SEQ_LM,
    },
}

TEXT_GEN_MODELS = {
    "opt": {
        "class": AutoModelForCausalLM,
        "processor": AutoTokenizer,
        "task_type": TaskType.CAUSAL_LM,
    },
    "rwkv": {
        "class": AutoModelForCausalLM,
        "processor": AutoTokenizer,
        "task_type": TaskType.CAUSAL_LM,
    },
    "gpt_neox": {
        "class": AutoModelForCausalLM,
        "processor": AutoTokenizer,
        "task_type": TaskType.CAUSAL_LM,
    },
    "RefinedWebModel": {
        "class": AutoModelForCausalLM,
        "processor": AutoTokenizer,
        "task_type": TaskType.CAUSAL_LM,
    },
    "llama": {
        "class": AutoModelForCausalLM,
        "processor": LlamaTokenizer,
        "task_type": TaskType.CAUSAL_LM,
    },
}
