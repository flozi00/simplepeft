import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from ..train.model import lightningmodel

import warnings

warnings.simplefilter("ignore")


def start_training(model, processor, dloader, PEFT_MODEL, LR: float, model_conf: dict):
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
        accumulate_grad_batches=1,
        callbacks=[lr_monitor],
        gradient_clip_val=0.5,
    )
    trainer.fit(model=plmodel, train_dataloaders=dloader)
