import torch
from chat_data import get_chat_dataset
from simplepeft.data import get_dataloader
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks
from datasets import Dataset
from retnet.configuration_retnet import RetNetConfig
from retnet.modeling_retnet import RetNetModel, RetNetModelWithLMHead
from transformers import AutoTokenizer

BATCH_SIZE = 4
PEFT_MODEL = "RetNet-small-german-assistant-v1"
TASK = Tasks.TEXT_GEN
LR = 1e-4

ASSISTANT_PREFIX = "### Assistant:"
USER_PREFIX = "### User:"

SEQ_LEN = 512


def main():
    RetNetConfig.register_for_auto_class()
    RetNetModel.register_for_auto_class("AutoModel")
    RetNetModelWithLMHead.register_for_auto_class("AutoModelForCausalLM")

    processor = AutoTokenizer.from_pretrained("philschmid/instruct-igel-001")
    processor.model_max_length = SEQ_LEN
    processor.pad_token = processor.eos_token

    try:
        model = RetNetModelWithLMHead.from_pretrained(PEFT_MODEL)
    except:
        model = RetNetModelWithLMHead(
            RetNetConfig(
                num_layers=12,
                hidden_size=1024,
                qk_dim=1280,
                v_dim=2560,
                ffn_proj_size=2560,
                forward_impl="parallel",  # parallel , recurrent , chunkwise
                pad_token_id=processor.pad_token_id,
                eos_token_id=processor.eos_token_id,
                vocab_size=len(processor),
            )
        )
    model = model.train().cuda()

    def eval_fun():
        model.eval()
        prompt = f"{USER_PREFIX} Wer ist aktuell der deutsche Bundeskanzler ?\n\n\n{ASSISTANT_PREFIX}"
        inputs = processor(prompt, return_tensors="pt").input_ids.cuda()

        # Generate
        with torch.inference_mode():
            generate_ids = model.generate(
                input_ids=inputs, max_new_tokens=128, parallel_compute_prompt=False
            )
        decoded = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(decoded)
        model.train()

    ds: Dataset = get_chat_dataset()

    def edit_special_tokens(example):
        example["conversations"] = example["conversations"].replace(
            "<|endoftext|>", "\n\n\n"
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
        max_input_length=SEQ_LEN,
        text_key="conversations",
    )

    # start training
    start_training(
        model=model,
        processor=processor,
        dloader=dloader,
        PEFT_MODEL=PEFT_MODEL,
        LR=LR,
        callback=eval_fun,
    )


if __name__ == "__main__":
    main()
