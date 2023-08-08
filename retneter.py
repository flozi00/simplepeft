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
LR = 3e-5

ASSISTANT_PREFIX = "### Assistant:"
USER_PREFIX = "### User:"

SEQ_LEN = 512


def main():
    RetNetConfig.register_for_auto_class()
    RetNetModel.register_for_auto_class("AutoModel")
    RetNetModelWithLMHead.register_for_auto_class("AutoModelForCausalLM")

    processor = AutoTokenizer.from_pretrained("philschmid/instruct-igel-001")
    processor.model_max_length = SEQ_LEN
    processor.add_special_tokens({"pad_token": "<|endoftext|>"})
    processor.pad_token = processor.eos_token

    try:
        model = RetNetModelWithLMHead.from_pretrained(PEFT_MODEL)
    except:
        hidden_qk = 1024
        model = RetNetModelWithLMHead(
            RetNetConfig(
                num_layers=24,
                num_heads=8,
                hidden_size=hidden_qk,
                qk_dim=hidden_qk,
                v_dim=hidden_qk * 2,
                ffn_proj_size=hidden_qk * 2,
                forward_impl="parallel",  # parallel , recurrent , chunkwise
                pad_token_id=processor.pad_token_id,
                eos_token_id=processor.eos_token_id,
                vocab_size=len(processor),
            )
        )
    model = model.train().cuda()

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")

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
