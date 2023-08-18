from chat_data import get_chat_dataset
from simplepeft.data import get_dataloader
from simplepeft.models import get_model
from simplepeft.utils import Tasks
from datasets import Dataset
from peft import PeftModelForCausalLM
import simplepeft.train.train

simplepeft.train.train.ACCUMULATION_STEPS = 4

BATCH_SIZE = 1
BASE_MODEL = "OpenBuddy/openbuddy-llama2-13b-v8.1-fp16"
PEFT_MODEL = "Llama-2-13b-german-assistant-v6"
TASK = Tasks.TEXT_GEN
LR = 1e-5

ROPE_FAKTOR = 1
SEQ_LENGTH = 4096

ASSISTANT_PREFIX = " ### Assistant: "
USER_PREFIX = " ### User: "


def main():
    # load model, processor and model_conf by using the get_model function
    model, processor, model_conf = get_model(
        task=TASK,
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL,
        use_peft=True,  # type: ignore
        use_py_flash=False,
    )

    model: PeftModelForCausalLM = model

    def eval_fun():
        model.eval()
        prompt = f"{USER_PREFIX} Wer ist aktuell der deutsche Bundeskanzler ?{processor.eos_token}{ASSISTANT_PREFIX}"
        inputs = processor(prompt, return_tensors="pt")

        # Generate
        generate_ids = model.generate(**inputs, max_new_tokens=128)
        decoded = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(decoded)
        model.train()

    # model.config.rope_scaling = {"type": "dynamic", "factor": ROPE_FAKTOR}

    ds: Dataset = get_chat_dataset()

    def edit_special_tokens(example):
        example["conversations"] = example["conversations"].replace(
            "<|endoftext|>", processor.eos_token
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
        max_input_length=SEQ_LENGTH,
        text_key="conversations",
    )

    # start training
    simplepeft.train.train.start_training(
        model=model,
        processor=processor,
        dloader=dloader,
        PEFT_MODEL=PEFT_MODEL,
        LR=LR,
        callback=eval_fun,
        kbit=model_conf.get("kbit", True),
    )


if __name__ == "__main__":
    main()
