from auto_gptq import AutoGPTQForCausalLM, get_gptq_peft_model
from auto_gptq.utils.peft_utils import GPTQLoraConfig
from peft import TaskType

from chat_data import get_chat_dataset
from simplepeft.data import get_dataloader
from simplepeft.train.train import start_training
from simplepeft.utils import Tasks
from datasets import Dataset
from transformers import AutoTokenizer
from optimum.bettertransformer import BetterTransformer

BATCH_SIZE = 1
BASE_MODEL = "OpenAssistant/llama2-13b-orca-8k-3319"
PEFT_MODEL = "llama2-13b-german-assistant-v3"
TASK = Tasks.TEXT_GEN
LR = 3e-5

ROPE_FAKTOR = 1


def main():
    # creating model
    peft_config = GPTQLoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoGPTQForCausalLM.from_quantized(
        BASE_MODEL,
        use_triton=False,
        warmup_triton=False,
        trainable=True,
        inject_fused_attention=False,
        inject_fused_mlp=False
    )
    model = BetterTransformer.transform(model)
    model = get_gptq_peft_model(model, peft_config=peft_config, auto_find_all_linears=True, train_mode=True)

    cv_data: Dataset = get_chat_dataset()

    def edit_special_tokens(example):
        example["conversations"] = example["conversations"].replace(
            "<|endoftext|>", tokenizer.eos_token
        )

        return example

    cv_data = cv_data.map(edit_special_tokens)

    # get the dataloader and define config for data loading and transformation
    dloader = get_dataloader(
        task=TASK,  # type: ignore
        processor=tokenizer,
        datas=cv_data,
        BATCH_SIZE=BATCH_SIZE,
        max_input_length=4096 * ROPE_FAKTOR,
        text_key="conversations",
    )

    # start training
    start_training(
        model=model,
        processor=tokenizer,
        dloader=dloader,
        PEFT_MODEL=PEFT_MODEL,
        LR=LR,
        model_conf={"is8bit": False},
        batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    main()