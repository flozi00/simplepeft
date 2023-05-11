from simplepeft.models import get_model
from transformers.pipelines import AutomaticSpeechRecognitionPipeline
import gradio as gr
from simplepeft.utils import Tasks

MODELNAME = "flozi00/whisper-large-german-cv13-sounds"

model, processor, model_conf = get_model(
    task=Tasks.ASR,
    model_name=MODELNAME,
    use_peft=False,
)

# model = model.half()
# model.push_to_hub(MODELNAME)

pipe = AutomaticSpeechRecognitionPipeline(
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
)


def transcribe(audio):
    text = pipe(audio)["text"]
    return text


gr.Interface(
    fn=transcribe, inputs=gr.Audio(source="upload", type="filepath"), outputs="text"
).launch()
