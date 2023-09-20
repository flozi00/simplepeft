from chat_data import get_chat_dataset
from simplepeft.data import get_dataloader
from datasets import Dataset
import simplepeft.train.train
from simplepeft.utils import Tasks

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from moduleformer import (
    ModuleFormerForCausalLM,
    ModuleFormerConfig,
    ModuleFormerForSequenceClassification,
)

AutoConfig.register("moduleformer", ModuleFormerConfig)
AutoModelForCausalLM.register(ModuleFormerConfig, ModuleFormerForCausalLM)
AutoModelForSequenceClassification.register(
    ModuleFormerConfig, ModuleFormerForSequenceClassification
)

simplepeft.train.train.ACCUMULATION_STEPS = 4

BATCH_SIZE = 1
LR = 1e-5

SEQ_LENGTH = 2048

ASSISTANT_PREFIX = " ### Assistant:"
USER_PREFIX = " ### User:"
END_SUFFIX = "</s>"
TASK = Tasks.TEXT_GEN

processor = AutoTokenizer.from_pretrained("ibm/MoLM-700M-8B")
model = AutoModelForCausalLM.from_pretrained("ibm/MoLM-700M-8B").cuda()

processor.pad_token = processor.eos_token


def main():
    ds: Dataset = get_chat_dataset()

    def eval_fun():
        model.eval()
        prompt = f"{USER_PREFIX} Wer ist aktuell der deutsche Bundeskanzler ?{processor.eos_token} {ASSISTANT_PREFIX}"
        inputs = processor(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        generate_ids = model.generate(**inputs, max_new_tokens=32)
        decoded = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(decoded)
        model.train()

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
        PEFT_MODEL="MoLM-700M-8B-german",
        LR=LR,
        callback=eval_fun,
    )


if __name__ == "__main__":
    main()
