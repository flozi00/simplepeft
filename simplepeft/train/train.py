from accelerate import Accelerator
import warnings
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.adam import Adam

warnings.simplefilter("ignore")


def start_training(model, processor, dloader, PEFT_MODEL, LR: float, callback=None):
    accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=4)
    accelerator.init_trackers("huggingface")

    try:
        from bitsandbytes.optim import PagedAdamW32bit

        optim = PagedAdamW32bit(model.parameters(), lr=LR)
    except Exception:
        optim = Adam(model.parameters(), lr=LR)

    scheduler = ExponentialLR(optim, gamma=0.95)
    model, optim, dloader, scheduler = accelerator.prepare(
        model, optim, dloader, scheduler
    )

    if callback is not None:
        callback()

    model.train()
    index = 1
    for data in (pbar := tqdm(dloader)):
        if index % 100 == 0:
            if callback is not None:
                callback()
            model.save_pretrained(
                PEFT_MODEL,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=True,
            )
            processor.save_pretrained(PEFT_MODEL)

        with accelerator.accumulate(model):
            output = model(return_dict=True, **data)
            loss = output.loss
            accelerator.backward(loss)

            pbar.set_description(
                f"Loss: {loss}",
                refresh=True,
            )

            accelerator.log({"training_loss": loss}, step=index - 1)

            optim.step()
            scheduler.step()
            optim.zero_grad()

        index += 1
