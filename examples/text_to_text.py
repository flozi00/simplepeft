import datasets
from simplepeft.data.main import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks

BATCH_SIZE = 24
BASE_MODEL = "t5-large"
PEFT_MODEL = "t5-large-german-instructions"
TASK = Tasks.Text2Text
LR = 1e-3


def map_to_ds(example):
    example[
        "prompt"
    ] = f'prompt: {example["instruction"]} </s> context: {example["context"]}'
    example["target"] = example["response"]

    return example


def get_dataset():
    ds_batch = []
    for x in ["de", "en", "es", "fr"]:
        ds = datasets.load_dataset(
            "argilla/databricks-dolly-15k-curated-multilingual", split=x
        )
        ds = ds.filter(
            lambda x: x["category"]
            in [
                "information_extraction",
                "closed_qa",
            ]
        )

        ds = ds.map(map_to_ds, remove_columns=ds.column_names)
        ds_batch.append(ds)

    ds = datasets.concatenate_datasets(ds_batch)

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
        max_input_length=1024,
        max_output_length=256,
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
