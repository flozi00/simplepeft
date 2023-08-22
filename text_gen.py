from chat_data import get_chat_dataset
from simplepeft.data import get_dataloader
from simplepeft.models import get_model
from simplepeft.utils import Tasks
from datasets import Dataset
from peft import PeftModelForCausalLM
import simplepeft.train.train

simplepeft.train.train.ACCUMULATION_STEPS = 16

BATCH_SIZE = 4
BASE_MODEL = "flozi00/Llama-2-7b-german-assistant-v3"
PEFT_MODEL = "Llama-2-7b-german-assistant-v4"
TASK = Tasks.TEXT_GEN
LR = 1e-5

SEQ_LENGTH = 4096

ASSISTANT_PREFIX = " ### Assistant: "
USER_PREFIX = " ### User: "


def main():
    ds: Dataset = get_chat_dataset().filter(lambda x: x["mode"] == "fine-tune")

    def combine_strings(strings):
        result = []
        current_string = strings[0]
        for string in strings[1:]:
            if len(current_string + string) <= SEQ_LENGTH * 3:
                current_string += string
            else:
                result.append(current_string)
                current_string = string
        result.append(current_string)
        return result

    new_conversations = combine_strings(ds["conversations"])
    print(len(ds["conversations"]), "-->", len(new_conversations))

    ds = Dataset.from_dict({"conversations": new_conversations})

    # load model, processor and model_conf by using the get_model function
    model, processor, model_conf = get_model(
        task=TASK,
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL,
        use_peft=True,  # type: ignore
        use_py_flash=True,
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
        # callback=eval_fun,
        kbit=model_conf.get("kbit", True),
    )


if __name__ == "__main__":
    main()
