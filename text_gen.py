from chat_data import get_chat_dataset
from simplepeft.data import get_dataloader
from simplepeft.models import get_model
from simplepeft.utils import Tasks
from datasets import Dataset
from peft import PeftModelForCausalLM
import simplepeft.train.train

simplepeft.train.train.ACCUMULATION_STEPS = 4

BATCH_SIZE = 1
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
PEFT_MODEL_BASE = "Mistral-7B-german-moe-v1-"
TASK = Tasks.TEXT_GEN
LR = 1e-5

SEQ_LENGTH = 4096

ASSISTANT_PREFIX = " ### Assistant:"
USER_PREFIX = " ### User:"
END_SUFFIX = "</s>"


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


def main():
    ds: Dataset = get_chat_dataset()
    labels = ds.unique("labels")

    # load model, processor by using the get_model function
    model, processor = get_model(
        task=TASK,
        model_name=BASE_MODEL,
        peft_name=PEFT_MODEL_BASE,
        use_peft=True,
        use_py_flash=False,
        use_flash_v2=True,
        use_bnb=True,
        lora_depth=32,
    )

    model: PeftModelForCausalLM = model

    def eval_fun():
        model.eval()
        prompt = f"{USER_PREFIX} Wer ist aktuell der deutsche Bundeskanzler ?{processor.eos_token} {ASSISTANT_PREFIX}"
        inputs = processor(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        generate_ids = model.generate(**inputs, max_new_tokens=32)
        decoded = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        print(decoded)
        model.train()

    for label in labels:
        PEFT_MODEL = PEFT_MODEL_BASE + label

        ds_part = ds.filter(lambda x: x["labels"] == label)

        ds_part = Dataset.from_dict(
            {
                "conversations": combine_strings(ds_part["conversations"]),
            }
        )

        # get the dataloader and define config for data loading and transformation
        dloader = get_dataloader(
            task=TASK,
            processor=processor,
            datas=ds_part,
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
        )


if __name__ == "__main__":
    main()
