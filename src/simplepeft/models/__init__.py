from ..models.speech import SPEECH_MODELS, TTS_MODELS
from ..models.text import TEXT_GEN_MODELS, TEXT_TEXT_MODELS
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
import torch
from transformers import AutoConfig

from ..utils import Tasks

try:
    import bitsandbytes as bnb

    bnb_try = bnb.optim  # only for linter
    bnb_available = True
except ImportError:
    bnb_available = False


def get_model(task: str, model_name: str, peft_name: str = None, use_peft=True):
    """Get the ready to use pef model, processor and model config
    Args: task (str): Task for the model
        model_name (str): Name of the model
        peft_name (str): Name of the peft model. If None, then the model_name is used
    Returns:
        model (Model): The ready to use model
        processor (Processor): The processor to use with the model
        model_conf (dict): The model config"""
    if task == Tasks.ASR:
        list_to_check = list(SPEECH_MODELS.keys())
        list_to_use = SPEECH_MODELS
    elif task == Tasks.TEXT_GEN:
        list_to_check = list(TEXT_GEN_MODELS.keys())
        list_to_use = TEXT_GEN_MODELS
    elif task == Tasks.Text2Text:
        list_to_check = list(TEXT_TEXT_MODELS.keys())
        list_to_use = TEXT_TEXT_MODELS
    elif task == Tasks.TTS:
        list_to_check = list(TTS_MODELS.keys())
        list_to_use = TTS_MODELS

    # check if the model_name is a peft model, if True, get the base model name from the config
    # otherwise, dont do anything
    try:
        conf = LoraConfig.from_pretrained(model_name)
        peft_name = model_name
        model_name = conf.base_model_name_or_path
    except Exception:
        pass

    # get the config of the base model and extract the model type from it
    conf = AutoConfig.from_pretrained(model_name)
    model_type_by_config = conf.model_type

    # check if the model_type is in the list of models
    for model_type in list_to_check:
        if model_type.lower() in model_type_by_config.lower():
            # get the model config
            model_conf = list_to_use[model_type]
            bnb_compatible = (
                model_conf.get("8-bit") is True
                and bnb_available is True
                and use_peft is True
            )
            # load the pre-trained model and check if its 8-bit compatible
            model = model_conf.get("class").from_pretrained(
                model_name, load_in_8bit=bnb_compatible, device_map="auto"
            )

            # load the processor
            processor = model_conf.get("processor").from_pretrained(model_name)

            # check if the model_type is whisper or mctct and set the config accordingly
            if model_type == "whisper":
                model.config.forced_decoder_ids, model.config.suppress_tokens = (
                    None,
                    [],
                )
            elif model_type == "mctct":
                model.config.ctc_loss_reduction = "mean"

            try:
                if processor.pad_token is None:
                    processor.add_special_tokens({"pad_token": "[PAD]"})
                    model.resize_token_embeddings(len(processor))
            except:
                if processor.tokenizer.pad_token is None:
                    processor.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    model.resize_token_embeddings(len(processor.tokenizer))

            # check if peft_name is not None, if True, load the peft model
            if peft_name is not None:
                # check if the model is 8-bit compatible and prepare it for 8-bit training
                if bnb_compatible:
                    model = prepare_model_for_int8_training(
                        model,
                        output_embedding_layer_name=model_conf.get(
                            "output_embedding_layer_name", "lm_head"
                        ),
                    )

                # create the lora config
                peft_config = LoraConfig(
                    r=8,
                    lora_alpha=64,
                    target_modules=model_conf.get("target_modules"),
                    lora_dropout=0.0,
                    task_type=model_conf.get("task_type", None),
                    inference_mode=False,
                    modules_to_save=model_conf.get("modules_to_save", None),
                )

                # load the peft model if possible, otherwise, create it from the base model and the lora config
                try:
                    model = PeftModel.from_pretrained(
                        model,
                        peft_name,
                    )
                    if use_peft is False:
                        model = model.merge_and_unload()
                except Exception as e:
                    print(e)
                    if use_peft:
                        model = get_peft_model(model, peft_config)

            return model, processor, model_conf

    # if the model_type is not in the list of supported models, raise an error
    raise ValueError(
        f"Model type for {model_name} not found in {list_to_check}, missing {model_type_by_config}"
    )
