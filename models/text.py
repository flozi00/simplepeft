from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

TEXT_TEXT_MODELS = {
    "t5": {
        "class": AutoModelForSeq2SeqLM,
        "processor": AutoTokenizer,
        "8-bit": True,
        "target_modules": ["q", "v"],
        "task_type": "SEQ_2_SEQ_LM",
    },
}

TEXT_GEN_MODELS = {
    "opt": {
        "class": AutoModelForCausalLM,
        "processor": AutoTokenizer,
        "8-bit": False,
        "target_modules": [
            "k_proj",
            "v_proj",
            "q_proj",
            "out_proj",
            "fc1",
            "fc2",
        ],
        "task_type": "CAUSAL_LM",
    },
}
