import datasets
from simplepeft.data.main import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks

BATCH_SIZE = 1
BASE_MODEL = "facebook/opt-1.3b"
PEFT_MODEL = "opt-1.3b-german-lora-instructions"
TASK = Tasks.TEXT_GEN
LR = 1e-6


def add_prefix(example):
    example["text"] = (
        example["instruction"] + " " + example["input"] + " " + example["output"]
    )
    return example


def get_dataset():
    ds = datasets.load_dataset(
        "philschmid/translated_tasks_de_google_52k", split="train"
    )
    ds = ds.filter(lambda x: len(x["input"]) > 5)

    ds = ds.map(add_prefix)

    ds = ds.filter(lambda x: len(x["instruction"]) < 1024 * 3)

    return ds


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
        max_input_length=1024,
        text_key="text",
    )

    start_training(
        model=model, processor=processor, dloader=dloader, PEFT_MODEL=PEFT_MODEL, LR=LR
    )


if __name__ == "__main__":
    main()
