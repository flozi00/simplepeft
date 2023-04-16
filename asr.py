import datasets
from simplepeft.data.main import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks

BATCH_SIZE = 64
BASE_MODEL = "openai/whisper-large-v2"
PEFT_MODEL = "whisper-large-v2-german-lora-cv13"
TASK = Tasks.ASR
LR = 1e-5


def get_dataset():
    d_sets = datasets.load_dataset(
        "mozilla-foundation/common_voice_13_0", "de", split="train"
    )

    d_sets = d_sets.filter(lambda x: len(x["sentence"]) > 5)
    d_sets = d_sets.filter(lambda x: x["down_votes"] <= 0 and x["up_votes"] >= 2)

    d_sets = d_sets.cast_column("audio", datasets.features.Audio(sampling_rate=16000))
    d_sets = d_sets.shuffle(seed=48)
    d_sets = d_sets.with_format("torch")

    return d_sets


def main():
    model, processor = get_model(
        task=TASK,
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL,
    )

    cv_data = get_dataset()

    dloader = get_dataloader(
        task=TASK,
        processor=processor,
        datas=cv_data,
        BATCH_SIZE=BATCH_SIZE,
    )

    start_training(
        model=model, processor=processor, dloader=dloader, PEFT_MODEL=PEFT_MODEL, LR=LR
    )


if __name__ == "__main__":
    main()
