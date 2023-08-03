from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from chat_data import get_chat_dataset
import fire

TESTS = [
    "<|prompter|>Können Sie das Rätsel des umweltfreundlichen Einkaufs lösen? Berücksichtigen Sie die Korrelation zwischen dem Preis eines Produkts und dessen Umweltauswirkungen. Aber Vorsicht, nicht alle teuren Artikel sind notwendigerweise umweltverträglich. Achten Sie auf irreführende Marketingtaktiken und Greenwashing in der Industrie. Können Sie die wahren umweltfreundlichen Optionen entschlüsseln, ohne dabei pleite zu gehen? Lass uns deine Denkfähigkeiten auf die Probe stellen. <|endoftext|><|assistant|>",
    "<|prompter|>Erstellen Sie eine Liste von 5 Arten von Dinosauriern. <|endoftext|><|assistant|>",
    "<|prompter|>Wer bist denn du ?<|endoftext|><|assistant|>",
]


def convert_model(BITS, model):
    BITS = int(BITS)

    pretrained_model_dir = model
    quantized_model_dir = f"{pretrained_model_dir.split('/')[-1]}-{BITS}bit-autogptq"

    cv_data = get_chat_dataset().shuffle(seed=42).select(range(50))

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_dir, use_fast=True, legacy=False
    )
    tokenizer.push_to_hub(repo_id=quantized_model_dir)
    examples = [tokenizer(data["conversations"]) for data in cv_data]

    quantize_config = BaseQuantizeConfig(
        bits=BITS,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(
        pretrained_model_dir,
        quantize_config,
        max_memory={0: "16GB", "cpu": "128GB"},
        trust_remote_code=True,
    )

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(examples)

    model.cpu()

    # save quantized model
    model.push_to_hub(repo_id=quantized_model_dir, save_dir=quantized_model_dir)


if __name__ == "__main__":
    fire.Fire(convert_model)
