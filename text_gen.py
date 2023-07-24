from chat_data import get_chat_dataset
from simplepeft.data import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks
from datasets import Dataset

BATCH_SIZE = 3
BASE_MODEL = "flozi00/Llama-2-7b-german-assistant-v1"
PEFT_MODEL = "Llama-2-7b-german-assistant-v2"
TASK = Tasks.TEXT_GEN
LR = 1e-5

ROPE_FAKTOR = 2


def main():
    # load model, processor and model_conf by using the get_model function
    model, processor, model_conf = get_model(
        task=TASK, model_name=BASE_MODEL, peft_name=PEFT_MODEL, use_peft=True  # type: ignore
    )

    # model.config.rope_scaling = {"type": "dynamic", "factor": ROPE_FAKTOR}

    cv_data: Dataset = get_chat_dataset()

    def edit_special_tokens(example):
        example["conversations"] = example["conversations"].replace(
            "<|endoftext|>", processor.eos_token
        )

        return example

    cv_data = cv_data.map(edit_special_tokens)

    # get the dataloader and define config for data loading and transformation
    dloader = get_dataloader(
        task=TASK,  # type: ignore
        processor=processor,
        datas=cv_data,
        BATCH_SIZE=BATCH_SIZE,
        max_input_length=512 * 4 * ROPE_FAKTOR,
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
        batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    main()
