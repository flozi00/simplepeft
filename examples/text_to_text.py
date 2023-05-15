import datasets
from simplepeft.data.main import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks

BATCH_SIZE = 1
BASE_MODEL = "t5-large"
PEFT_MODEL = "t5-large-german-lora-instructions"
TASK = Tasks.Text2Text
LR = 1e-5


# generate an instruction dataset by using the instruction as prefix for the input
def add_prefix(example):
    example["prompt"] = example["instruction"] + "\n" + example["input"]
    example["target"] = example["output"]
    return example


def map_to_ds(example):
    example["prompt"] = example["instruction"] + "\n" + example["context"]
    example["target"] = example["response"]

    return example


def get_dataset():
    ds = datasets.load_dataset(
        "philschmid/translated_tasks_de_google_52k", split="train"
    )
    ds = ds.map(add_prefix, remove_columns=ds.column_names)

    ds2 = datasets.load_dataset(
        "argilla/databricks-dolly-15k-curated-multilingual", split="de"
    )
    ds2 = ds2.map(map_to_ds, remove_columns=ds2.column_names)

    ds = datasets.concatenate_datasets([ds, ds2])

    ds = ds.filter(lambda x: len(x["prompt"]) > 128)

    return ds


def main():
    # load model, processor and model_conf by using the get_model function
    model, processor, model_conf = get_model(
        task=TASK,
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL,
    )

    cv_data = get_dataset()

    # get the dataloader and define config for data loading and transformation
    dloader = get_dataloader(
        task=TASK,
        processor=processor,
        datas=cv_data,
        BATCH_SIZE=BATCH_SIZE,
        source_key="prompt",
        target_key="target",
        max_input_length=4096,
        max_output_length=512,
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
