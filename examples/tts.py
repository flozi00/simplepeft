import datasets
from simplepeft.data.main import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks

BATCH_SIZE = 12
BASE_MODEL = "microsoft/speecht5_tts"
PEFT_MODEL = "speecht5_tts-german-lora-cv-simplepeft"
TASK = Tasks.TTS
LR = 1e-4

import torch
from speechbrain.pretrained import EncoderClassifier

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def get_dataset():
    d_sets = datasets.load_dataset("common_voice", "de", split="train")

    d_sets = d_sets.cast_column("audio", datasets.features.Audio(sampling_rate=16000))
    d_sets = d_sets.shuffle(seed=48)

    print(d_sets)

    return d_sets


cv_data = get_dataset()


def main():
    model, processor, model_conf = get_model(
        task=TASK,
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL,
    )

    spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    speaker_model = EncoderClassifier.from_hparams(
        source=spk_model_name,
        run_opts={"device": device},
        savedir="pretrained_models/speaker_model",
    )

    dloader = get_dataloader(
        task=TASK,
        processor=processor,
        datas=cv_data,
        BATCH_SIZE=BATCH_SIZE,
        speaker_model=speaker_model,
        text_key="sentence",
    )

    start_training(
        model=model,
        processor=processor,
        dloader=dloader,
        PEFT_MODEL=PEFT_MODEL,
        LR=LR,
        model_conf=model_conf,
    )


if __name__ == "__main__":
    main()
