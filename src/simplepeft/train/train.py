import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from ..train.model import lightningmodel

import warnings

warnings.simplefilter("ignore")


def start_training(model, processor, dloader, PEFT_MODEL, LR: float, model_conf: dict):
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
    plmodel = lightningmodel(
        model_name=PEFT_MODEL,
        model=model,
        processor=processor,
        lr=LR,
    )

    _logger = WandbLogger(project="huggingface", name=PEFT_MODEL)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        logger=_logger,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=1,
        precision=model_conf.get("precision", 32),
        accumulate_grad_batches=4,
        callbacks=[lr_monitor],
        gradient_clip_val=0.5,
    )
    trainer.fit(model=plmodel, train_dataloaders=dloader)
