import torch
from ..models.speech import SPEECH_MODELS, TTS_MODELS
from ..models.text import TEXT_GEN_MODELS, TEXT_TEXT_MODELS
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)
from transformers import AutoConfig, BitsAndBytesConfig
from ..utils import Tasks
from optimum.bettertransformer import BetterTransformer


try:
    import bitsandbytes as bnb

    bnb_try = bnb.optim  # only for linter
    bnb_available = True
except ImportError:
    bnb_available = False


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    )
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # if not is_gptq_quantized:
    # cast all non INT8 parameters to fp32
    #    for param in model.parameters():
    #        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
    #            param.data = param.data.to(torch.float32)

    if (loaded_in_kbit or is_gptq_quantized) and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


def get_model(
    task: str,
    model_name: str,
    peft_name: str = None,
    use_peft=True,
    push_to_hub=False,
    processor_name: str = None,
    use_py_flash=True,
    use_bnb=True,
    lora_depth=128,
):
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
        lora_conf = LoraConfig.from_pretrained(pretrained_model_name_or_path=model_name)
        peft_name = model_name
        model_name = lora_conf.base_model_name_or_path
    except Exception:
        pass

    # get the config of the base model and extract the model type from it
    conf = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
    )
    model_type_by_config = conf.model_type
    kwargs = {}
    # check if the model_type is in the list of models
    for model_type in list_to_check:
        if model_type.lower() == model_type_by_config.lower():
            # get the model config
            model_conf = list_to_use[model_type]
            bnb_compatible = (
                bnb_available is True
                and use_peft is True
                and push_to_hub is False
                and use_bnb is True
            )

            try:
                processor = model_conf.get("processor").from_pretrained(
                    peft_name, legacy=False
                )
            except:
                # load the processor
                processor = model_conf.get("processor").from_pretrained(
                    model_name if processor_name is None else processor_name,
                    legacy=False,
                )

            if model_conf.get("config", None) is not None:
                print("Loading specific config")
                conf = model_conf.get("config").from_pretrained(
                    model_name,
                )

            # check if the model_type is whisper or mctct and set the config accordingly
            if model_type == "whisper":
                conf.forced_decoder_ids, conf.suppress_tokens = (
                    None,
                    [],
                )
            elif model_type in ["wav2vec2", "mctct"]:
                conf.update(
                    {
                        "feat_proj_dropout": 0.01,
                        "attention_dropout": 0.01,
                        "hidden_dropout": 0.01,
                        "final_dropout": 0.01,
                        "mask_time_prob": 0,
                        "mask_time_length": 0,
                        "mask_feature_prob": 0,
                        "mask_feature_length": 0,
                        "gradient_checkpointing": True,
                        "layerdrop": 0,
                        "ctc_loss_reduction": "mean",
                        "pad_token_id": processor.tokenizer.pad_token_id,
                        "vocab_size": len(processor.tokenizer),
                        "activation_dropout": 0.01,
                    }
                )
                kwargs["ignore_mismatched_sizes"] = True

            if bnb_compatible:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="fp4",
                )

            # load the pre-trained model and check if its 8-bit compatible
            model = model_conf.get("class").from_pretrained(
                model_name,
                config=conf,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                **kwargs,
            )

            if use_py_flash is True:
                try:
                    model = BetterTransformer.transform(model)
                    print("converted to Bettertransformer")
                except Exception as e:
                    print(e)

            try:
                if processor.pad_token is None:
                    processor.pad_token = processor.eos_token
                    print("Setting pad token to eos token")
            except:
                if processor.tokenizer.pad_token is None:
                    processor.tokenizer.pad_token = processor.tokenizer.eos_token
                    print("Setting pad token to eos token")

            # check if peft_name is not None, if True, load the peft model
            if peft_name is not None:
                # check if the model is 8-bit compatible and prepare it for 8-bit training
                if bnb_compatible:
                    print("Preparing model for K-bit training")
                    try:
                        model = prepare_model_for_kbit_training(
                            model, use_gradient_checkpointing=True
                        )
                    except Exception as e:
                        print(e)
                        model = prepare_model_for_kbit_training(
                            model, use_gradient_checkpointing=False
                        )

                def find_all_linear_names(model):
                    lora_module_names = set()
                    for name, module in model.named_modules():
                        if (
                            isinstance(module, bnb.nn.Linear4bit)
                            or isinstance(module, bnb.nn.Linear8bitLt)
                            or isinstance(module, torch.nn.Linear)
                        ):
                            names = name.split(".")
                            lora_module_names.add(
                                names[0] if len(names) == 1 else names[-1]
                            )

                    if "lm_head" in lora_module_names:  # needed for 16-bit
                        lora_module_names.remove("lm_head")
                    return list(lora_module_names)

                # create the lora config
                peft_config = LoraConfig(
                    r=lora_depth,
                    lora_alpha=lora_depth,
                    lora_dropout=0.01,
                    target_modules=find_all_linear_names(model),
                    task_type=model_conf.get("task_type", None),
                    inference_mode=False,
                    modules_to_save=model_conf.get("modules_to_save", None),
                )

                # load the peft model if possible, otherwise, create it from the base model and the lora config
                try:
                    model = PeftModel.from_pretrained(
                        model=model,
                        model_id=peft_name,
                        is_trainable=True,
                    )
                    print("Loaded peft model")
                    model.print_trainable_parameters()
                    if use_peft is False:
                        try:
                            model = model.merge_and_unload()
                            model = model.train()
                            for param in model.parameters():
                                param.requires_grad = True
                            print("Merged peft model to base model format")
                        except Exception as e:
                            print(e)
                except Exception as e:
                    print(e)
                    if use_peft:
                        print("Creating peft model")
                        model = get_peft_model(model=model, peft_config=peft_config)
                        model.print_trainable_parameters()
            else:
                peft_name = model_name

            if push_to_hub:
                PUSH_NAME = peft_name.split(sep="/")[-1]
                model.half()
                model = BetterTransformer.reverse(model)
                model.save_pretrained(PUSH_NAME, safe_serialization=True)
                processor.save_pretrained(PUSH_NAME)

                model.push_to_hub(PUSH_NAME, safe_serialization=True)
                processor.push_to_hub(PUSH_NAME)

            model_conf["kbit"] = bnb_compatible
            model_conf["is_peft"] = use_peft
            model_conf["peft"] = peft_config

            return model, processor, model_conf

    # if the model_type is not in the list of supported models, raise an error
    raise ValueError(
        f"Model type for {model_name} not found in {list_to_check}, missing {model_type_by_config}"
    )
