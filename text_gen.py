from chat_data import get_chat_dataset
from simplepeft.data import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks

BATCH_SIZE = 1
BASE_MODEL = "OpenBuddy/openbuddy-openllama-13b-v7-fp16"
PEFT_MODEL = "openllama-13b-german-assistant"
TASK = Tasks.TEXT_GEN
LR = 1e-4


def main():
    # load model, processor and model_conf by using the get_model function
    model, processor, model_conf = get_model(
        task=TASK, model_name=BASE_MODEL, peft_name=PEFT_MODEL, use_peft=True  # type: ignore
    )

    cv_data = get_chat_dataset()

    # get the dataloader and define config for data loading and transformation
    dloader = get_dataloader(
        task=TASK,  # type: ignore
        processor=processor,
        datas=cv_data,
        BATCH_SIZE=BATCH_SIZE,
        max_input_length=2048,
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
