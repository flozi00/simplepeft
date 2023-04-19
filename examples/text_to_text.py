import datasets
from simplepeft.data.main import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks

BATCH_SIZE = 32
BASE_MODEL = "t5-large"
PEFT_MODEL = "t5-large-german-lora-instructions"
TASK = Tasks.Text2Text
LR = 1e-5

# generate an instruction dataset by using the instruction as prefix for the input
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
        source_key="input",
        target_key="output",
        max_input_length=512,
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
