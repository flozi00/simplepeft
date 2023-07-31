from accelerate import Accelerator
import warnings
from tqdm.auto import tqdm
import time
import GPUtil
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.adam import Adam

warnings.simplefilter("ignore")

TEMP_LIMIT = 68


def start_training(
    model, processor, dloader, PEFT_MODEL, LR: float, model_conf: dict, batch_size: int
):
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers("huggingface")

    if model_conf["is8bit"]:
        from bitsandbytes.optim import PagedAdam

        optim = PagedAdam(model.parameters(), lr=LR)
    else:
        optim = Adam(model.parameters(), lr=LR)

    scheduler = ExponentialLR(optim, gamma=0.95)
    model, optim, dloader, scheduler = accelerator.prepare(
        model, optim, dloader, scheduler
    )

    model.train()
    index = 1
    for data in (pbar := tqdm(dloader)):
        if index % 100 == 0:
            model.save_pretrained(
                PEFT_MODEL,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )

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

        gpus = GPUtil.getGPUs()
        for gpu_num in range(len(gpus)):
            gpu = gpus[gpu_num]
            if gpu.temperature >= TEMP_LIMIT:
                faktor = int(gpu.temperature) - TEMP_LIMIT
                time.sleep(faktor * 5)
