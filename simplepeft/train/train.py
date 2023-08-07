from accelerate import Accelerator
import warnings
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.adam import Adam

warnings.simplefilter("ignore")

ACCUMULATION_STEPS = 16


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def start_training(model, processor, dloader, PEFT_MODEL, LR: float, callback=None):
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=ACCUMULATION_STEPS,
    )
    accelerator.init_trackers("huggingface")

    model.train()

    try:
        from bitsandbytes.optim import PagedAdamW32bit, PagedLion32bit

        optim = PagedLion32bit(model.parameters(), lr=LR)
    except Exception:
        optim = Adam(model.parameters(), lr=LR)

    scheduler = ExponentialLR(optim, gamma=0.99)
    model, optim, dloader, scheduler = accelerator.prepare(
        model, optim, dloader, scheduler, device_placement=[True, True, True, True]
    )

    if callback is not None:
        eval_ = callback()
        if eval_ is not None:
            accelerator.log({"eval_metric": eval_}, step=0)

    index = 1
    while True:
        for data in (pbar := tqdm(dloader)):
            if index % 1000 == 0:
                if callback is not None:
                    eval_ = callback()
                    if eval_ is not None:
                        accelerator.log({"eval_metric": eval_}, step=index - 1)
                model.save_pretrained(
                    PEFT_MODEL,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model),
                    safe_serialization=False,
                )
                processor.save_pretrained(PEFT_MODEL)

            optim.zero_grad()
            with accelerator.accumulate(model):
                output = model(return_dict=True, **data)
                loss = output.loss
                accelerator.backward(loss)

                pbar.set_description(
                    f"Loss: {loss} LR: {get_lr(optim.optimizer)}",
                    refresh=True,
                )
                if index % ACCUMULATION_STEPS == 0:
                    accelerator.log(
                        values={
                            "training_loss": loss,
                            "learning_rate": float(get_lr(optim.optimizer)),
                        },
                        step=int(index / ACCUMULATION_STEPS),
                    )
                if accelerator.sync_gradients:
                    accelerator.clip_grad_value_(model.parameters(), 0.9)
                optim.step()
                scheduler.step()

            index += 1
