from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModelForCTC,
    AutoProcessor,
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
)

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
        ],
        "output_embedding_layer_name": "proj_out",
        "precision": 16,
    },
    "wav2vec2": {
        "class": AutoModelForCTC,
        "processor": AutoProcessor,
        "8-bit": False,
        "target_modules": ["k_proj", "v_proj", "q_proj", "out_proj"],
        "precision": 16,
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
