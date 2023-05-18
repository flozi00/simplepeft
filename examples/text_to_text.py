import datasets
from simplepeft.data import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks

BATCH_SIZE = 24
BASE_MODEL = "facebook/bart-base"
PEFT_MODEL = "bart-base-cnn_dailymail"
TASK = Tasks.Text2Text
LR = 1e-5



def get_dataset():
    ds = datasets.load_dataset(
        "cnn_dailymail", "1.0.0", split="train"
    )

    return ds


def main():
    # load model, processor and model_conf by using the get_model function
    model, processor, model_conf = get_model(
        task=TASK,  # type: ignore
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL,
    )

    cv_data = get_dataset()

    # get the dataloader and define config for data loading and transformation
    dloader = get_dataloader(
        task=TASK,  # type: ignore
        processor=processor,
        datas=cv_data,
        BATCH_SIZE=BATCH_SIZE,
        source_key="article",
        target_key="highlights",
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
