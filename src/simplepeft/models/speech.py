from transformers import (
    MCTCTForCTC,
    MCTCTProcessor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModelForCTC,
    AutoProcessor,
)

SPEECH_MODELS = {
    "mctct": {
        "class": MCTCTForCTC,
        "processor": MCTCTProcessor,
        "8-bit": False,
        "target_modules": ["query", "value"],
    },
    "whisper": {
        "class": WhisperForConditionalGeneration,
        "processor": WhisperProcessor,
        "8-bit": True,
        "target_modules": [
            "k_proj",
            "v_proj",
            "q_proj",
            "out_proj",
            "fc1",
            "fc2",
        ],
        "output_embedding_layer_name": "proj_out",
        "precision": 16,
    },
    "wav2vec2": {
        "class": AutoModelForCTC,
        "processor": AutoProcessor,
        "8-bit": False,
        "target_modules": ["k_proj", "v_proj", "q_proj", "out_proj"],
    },
}
