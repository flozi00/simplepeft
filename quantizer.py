from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from chat_data import get_chat_dataset

BITS = 3

pretrained_model_dir = "flozi00/falcon-7b-german-assistant-v2"
quantized_model_dir = f"{pretrained_model_dir.split('/')[-1]}-{BITS}bit"

cv_data = get_chat_dataset().select(range(100))

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [tokenizer(data["conversations"]) for data in cv_data]

quantize_config = BaseQuantizeConfig(
    bits=BITS,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=True,  # set to False can significantly speed up inference but the perplexity may slightly bad
)

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(
    pretrained_model_dir, quantize_config, max_memory={"0": "12GB", "cpu": "64GB"}
)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples)

# save quantized model
model.push_to_hub(repo_id=quantized_model_dir, save_dir=quantized_model_dir)
