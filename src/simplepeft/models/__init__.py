from ..models.speech import SPEECH_MODELS
from ..models.text import TEXT_GEN_MODELS, TEXT_TEXT_MODELS
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
import torch
from transformers import AutoConfig

from ..utils import Tasks

try:

    bnb_available = True
except ImportError:
    bnb_available = False


def get_model(task: str, model_name: str, peft_name: str = None):
    if task == Tasks.ASR:
        list_to_check = list(SPEECH_MODELS.keys())
        list_to_use = SPEECH_MODELS
    elif task == Tasks.TEXT_GEN:
        list_to_check = list(TEXT_GEN_MODELS.keys())
        list_to_use = TEXT_GEN_MODELS
    elif task == Tasks.Text2Text:
        list_to_check = list(TEXT_TEXT_MODELS.keys())
        list_to_use = TEXT_TEXT_MODELS

    try:
        conf = LoraConfig.from_pretrained(model_name)
        peft_name = model_name
        model_name = conf.base_model_name_or_path
    except Exception:
        pass

    conf = AutoConfig.from_pretrained(model_name)
    model_type_by_config = conf.model_type

    for model_type in list_to_check:
        if model_type.lower() in model_type_by_config.lower():
            model = (
                list_to_use[model_type]
                .get("class")
                .from_pretrained(
                    model_name,
                    load_in_8bit=list_to_use[model_type].get("8-bit") and bnb_available,
                    device_map="auto",
                )
            )

            processor = (
                list_to_use[model_type].get("processor").from_pretrained(model_name)
            )

            if model_type == "whisper":
                model.config.forced_decoder_ids, model.config.suppress_tokens = (
                    None,
                    [],
                )
            elif model_type == "mctct":
                model.config.ctc_loss_reduction = "mean"

            if peft_name is not None:
                if list_to_use[model_type].get("8-bit") and bnb_available:
                    model = prepare_model_for_int8_training(
                        model,
                        output_embedding_layer_name=list_to_use[model_type].get(
                            "output_embedding_layer_name", "lm_head"
                        ),
                    )

                peft_config = LoraConfig(
                    r=8,
                    lora_alpha=64,
                    target_modules=list_to_use[model_type].get("target_modules"),
                    lora_dropout=0.0,
                    task_type=list_to_use[model_type].get("task_type", None),
                    inference_mode=False,
                )

                try:
                    model = PeftModel.from_pretrained(
                        model,
                        peft_name,
                    )
                except Exception as e:
                    print(e)
                    model = get_peft_model(model, peft_config)

            return model, processor

    raise ValueError(f"Model type for {model_name} not found in {list_to_check}")
