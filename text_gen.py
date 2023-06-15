import datasets
from simplepeft.data import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks

BATCH_SIZE = 1
BASE_MODEL = "OpenAssistant/falcon-7b-sft-mix-2000"
PEFT_MODEL = "falcon-7b-german"
TASK = Tasks.TEXT_GEN
LR = 1e-5

PROMPTER = "<|prompter|>"
BOT = "<|assistant|>"
END = "<|endoftext|>"


def get_dataset() -> datasets.Dataset:
    all_rows = []

    ds2 = datasets.load_dataset(
        "philschmid/translated_tasks_de_google_52k", split="train"
    )
    for row in ds2:
        if len(row["input"]) > 128:  # type: ignore
            all_rows.append(
                f'{PROMPTER}{row["instruction"]} {row["input"]} {END}{BOT}{row["output"]}{END}'  # type: ignore
            )

    ds2 = datasets.load_dataset(
        "argilla/databricks-dolly-15k-curated-multilingual", split="de"
    )
    for row in ds2:
        if len(row["context"]) > 128:  # type: ignore
            all_rows.append(
                f'{PROMPTER}{row["instruction"]} {row["context"]} {END}{BOT}{row["response"]}{END}'  # type: ignore
            )

    ds = datasets.Dataset.from_dict({"conversations": all_rows})

    ds3 = datasets.load_dataset(
        "flozi00/openassistant-oasst1-flattened-filtered", split="train"
    )

    ds = datasets.concatenate_datasets([ds, ds3])

    return ds


def main():
    # load model, processor and model_conf by using the get_model function
    model, processor, model_conf = get_model(
        task=TASK, model_name=BASE_MODEL, peft_name=PEFT_MODEL, use_peft=True  # type: ignore
    )

    cv_data = get_dataset()

    # get the dataloader and define config for data loading and transformation
    dloader = get_dataloader(
        task=TASK,  # type: ignore
        processor=processor,
        datas=cv_data,
        BATCH_SIZE=BATCH_SIZE,
        max_input_length=1024,
        text_key="conversations",
    )

    # start training
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
