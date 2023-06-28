from transformers import Wav2Vec2ForCTC, AutoProcessor

LANG = "deu"

model_id = "facebook/mms-1b-all"

processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

processor.tokenizer.set_target_lang(LANG)
model.load_adapter(LANG)

model.half()

processor.save_pretrained(
    f"mms-1b-{LANG}",
)
model.save_pretrained(f"mms-1b-{LANG}", safe_serialization=True)

processor.push_to_hub(
    f"mms-1b-{LANG}",
)
model.push_to_hub(f"mms-1b-{LANG}", safe_serialization=True)
