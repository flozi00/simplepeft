from chat_data import get_chat_dataset
from simplepeft.data import get_dataloader
from simplepeft.models import get_model
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks
from datasets import Dataset

BATCH_SIZE = 1
BASE_MODEL = "OpenBuddy/openbuddy-llama2-13b-v8.1-fp16"
PEFT_MODEL = "Llama-2-13b-german-assistant-v4"
TASK = Tasks.TEXT_GEN
LR = 3e-5

ROPE_FAKTOR = 1

ASSISTANT_PREFIX = "### Assistant:"
USER_PREFIX = "### User:"


def main():
    # load model, processor and model_conf by using the get_model function
    model, processor, model_conf = get_model(
        task=TASK, model_name=BASE_MODEL, peft_name=PEFT_MODEL, use_peft=True  # type: ignore
    )

    # model.config.rope_scaling = {"type": "dynamic", "factor": ROPE_FAKTOR}

    ds: Dataset = get_chat_dataset()

    def edit_special_tokens(example):
        example["conversations"] = example["conversations"].replace(
            "<|endoftext|>", "\n\n"
        )

        example["conversations"] = example["conversations"].replace(
            "<|prompter|>", USER_PREFIX
        )

        example["conversations"] = example["conversations"].replace(
            "<|assistant|>", ASSISTANT_PREFIX
        )

        return example

    ds = ds.map(edit_special_tokens)

    # get the dataloader and define config for data loading and transformation
    dloader = get_dataloader(
        task=TASK,  # type: ignore
        processor=processor,
        datas=ds,
        BATCH_SIZE=BATCH_SIZE,
        max_input_length=4096 * ROPE_FAKTOR,
        text_key="conversations",
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
