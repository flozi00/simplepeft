import datasets
from data.main import get_dataloader
from models import get_model
from train.train import start_training
from utils import Tasks

BATCH_SIZE = 64
BASE_MODEL = "GermanT5/t5-efficient-gc4-all-german-large-nl36"
PEFT_MODEL = "t5-large-german-lora-instructions"
TASK = Tasks.Text2Text
LR = 1e-5


def add_prefix(example):
    example["input"] = example["instruction"] + " " + example["input"]
    return example


def get_dataset():
    ds = datasets.load_dataset(
        "philschmid/translated_tasks_de_google_52k", split="train"
    )
    ds = ds.filter(lambda x: len(x["input"]) > 5)

    ds = ds.map(add_prefix)

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
        source_key="input",
        target_key="output",
        max_input_length=512,
        max_output_length=256,
    )

    start_training(
        model=model, processor=processor, dloader=dloader, PEFT_MODEL=PEFT_MODEL, LR=LR
    )


if __name__ == "__main__":
    main()
