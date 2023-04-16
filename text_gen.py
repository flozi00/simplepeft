import datasets
from src.data.main import get_dataloader
from src.models import get_model
from src.train.train import start_training
from src.utils import Tasks

BATCH_SIZE = 1
BASE_MODEL = "malteos/bloom-6b4-clp-german"
PEFT_MODEL = "bloom-6b4-german-lora-instructions"
TASK = Tasks.TEXT_GEN
LR = 1e-4


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

    processor.pad_token = processor.eos_token

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
