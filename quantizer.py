from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
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

    cv_data = get_chat_dataset().shuffle(seed=42).select(range(50))["conversations"]

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_dir, use_fast=True, legacy=False
    )
    tokenizer.push_to_hub(repo_id=quantized_model_dir)
    tokenizer.save_pretrained(quantized_model_dir)

    quantization = GPTQConfig(
        bits=BITS, dataset=cv_data, tokenizer=tokenizer, batch_size=1
    )

    model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        quantization_config=quantization,
        max_memory={0: "18GB", "cpu": "128GB"},
    )

    # save quantized model
    model.push_to_hub(
        repo_id=quantized_model_dir,
        save_dir=quantized_model_dir,
        safe_serialization=True,
    )


if __name__ == "__main__":
    fire.Fire(convert_model)
