import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lion_pytorch import Lion

from ..train.model import lightningmodel

import warnings

warnings.simplefilter("ignore")


def start_training(
    model,
    processor,
    dloader,
    PEFT_MODEL,
    LR: float,
    model_conf: dict,
):
    """Generating the training loop for the model, using pytorch lightning#
    Building the lightning module and the trainer for the model automatically

    Args:
        model (_type_): The model to train, from this library
        processor (_type_): The processor from the model
        dloader (_type_): The pytorch dataloader
        PEFT_MODEL (_type_): The name of the model to be saved as
        LR (float): The learning rate
        model_conf (dict): The model configuration from this library
    """
    strategy = "auto"

    if model_conf["is8bit"]:
        from bitsandbytes.optim import PagedLion

        optim = PagedLion
    else:
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optim = DeepSpeedCPUAdam
            strategy = "deepspeed_stage_2_offload"
        except:
            optim = Lion
    plmodel = lightningmodel(
        model_name=PEFT_MODEL,
        model=model,
        processor=processor,
        optim=optim,
        lr=LR,
        save_every_hours=1 if model_conf["is_peft"] else 6,
    )

    _logger = WandbLogger(project="huggingface", name=PEFT_MODEL)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        logger=_logger,
        log_every_n_steps=1,
        precision=16,
        accumulate_grad_batches=model_conf.get("gradient_accumulation", 1),
        callbacks=[lr_monitor],
        strategy=strategy,
        gradient_clip_val=0.7,
    )
    trainer.fit(model=plmodel, train_dataloaders=dloader)
