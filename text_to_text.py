from chat_data import get_chat_dataset
from simplepeft.data import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks

BATCH_SIZE = 4
BASE_MODEL = "allenai/led-large-16384"
PEFT_MODEL = "led-large-16384-german-assistent"
TASK = Tasks.Text2Text
LR = 1e-4


def main():
    ds = get_chat_dataset(T2T=True)

    # load model, processor and model_conf by using the get_model function
    model, processor, model_conf = get_model(
        task=TASK,  # type: ignore
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL,
        use_peft=True,
    )

    # get the dataloader and define config for data loading and transformation
    dloader = get_dataloader(
        task=TASK,  # type: ignore
        processor=processor,
        datas=ds,
        BATCH_SIZE=BATCH_SIZE,
        source_key="conversations",
        target_key="answers",
        max_input_length=4096 * 2,
        max_output_length=1024,
    )

    # start training
    start_training(
        model=model,
        processor=processor,
        dloader=dloader,
        PEFT_MODEL=PEFT_MODEL,
        LR=LR,
    )


if __name__ == "__main__":
    main()
