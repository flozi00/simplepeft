import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lion_pytorch import Lion

from ..train.model import lightningmodel

import warnings

from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning.pytorch.strategies import DeepSpeedStrategy

warnings.simplefilter("ignore")


def start_training(
    model,
    processor,
    dloader,
    PEFT_MODEL,
    LR: float,
    model_conf: dict,
    deepspeed: bool = False,
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
    plmodel = lightningmodel(
        model_name=PEFT_MODEL,
        model=model,
        processor=processor,
        optim=Lion if deepspeed is False else DeepSpeedCPUAdam,
        lr=LR,
    )

    _logger = WandbLogger(project="huggingface", name=PEFT_MODEL)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    strategy = DeepSpeedStrategy(
        offload_optimizer=True,
        offload_parameters=True,
        offload_optimizer_device="cpu",
        offload_params_device="cpu",
        zero_optimization=True,
        stage=2,
    )

    if deepspeed is False:
        strategy = "auto"

    trainer = pl.Trainer(
        logger=_logger,
        log_every_n_steps=1,
        precision=16,
        accumulate_grad_batches=model_conf.get("gradient_accumulation", 1),
        callbacks=[lr_monitor],
        gradient_clip_val=0.5,
        strategy=strategy,
    )
    trainer.fit(model=plmodel, train_dataloaders=dloader)
