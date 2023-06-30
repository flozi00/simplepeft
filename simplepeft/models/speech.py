from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModelForCTC,
    AutoProcessor,
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
    Wav2Vec2ForCTC,
)

Wav2Vec2ForCTC._no_split_modules = []

SPEECH_MODELS = {
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
        # "modules_to_save": ["embed_tokens"],
        "output_embedding_layer_name": "proj_out",
        "precision": 16,
    },
    "wav2vec2": {
        "class": Wav2Vec2ForCTC,
        "processor": AutoProcessor,
        "8-bit": False,
        "target_modules": [
            "k_proj",
            "v_proj",
            "q_proj",
            "out_proj",
            "intermediate_dense",
            "output_dense",
        ],
        "precision": 16,
        "modules_to_save": ["lm_head"],
        "gradient_accumulation": 1,
    },
}

TTS_MODELS = {
    "speecht5": {
        "class": SpeechT5ForTextToSpeech,
        "processor": SpeechT5Processor,
        "8-bit": False,
        "target_modules": [
            "k_proj",
            "v_proj",
            "q_proj",
            "out_proj",
        ],
        "precision": 16,
    },
}
