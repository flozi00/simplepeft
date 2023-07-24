from lion_pytorch import Lion
from accelerate import Accelerator
import warnings
from tqdm.auto import tqdm
import time
import GPUtil
from torch.optim.lr_scheduler import ExponentialLR

warnings.simplefilter("ignore")

TEMP_LIMIT = 72
ACCUMULATION_STEPS = 2


def start_training(
    model, processor, dloader, PEFT_MODEL, LR: float, model_conf: dict, batch_size: int
):
    accelerator = Accelerator(
        gradient_accumulation_steps=ACCUMULATION_STEPS, log_with="wandb"
    )
    device = accelerator.device
    accelerator.init_trackers("huggingface")

    if model_conf["is8bit"]:
        from bitsandbytes.optim import PagedAdamW8bit

        optim = PagedAdamW8bit(model.parameters(), lr=LR)
    else:
        optim = Lion(model.parameters(), lr=LR)

    scheduler = ExponentialLR(optim, gamma=0.98)
    model, optim, dloader, scheduler = accelerator.prepare(
        model, optim, dloader, scheduler
    )

    model.train()
    index = 0
    for data in (pbar := tqdm(dloader)):
        with accelerator.accumulate(model):
            index += 1
            if index % 100 == 0:
                model.save_pretrained(
                    PEFT_MODEL,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model),
                )
            optim.zero_grad()
            output = model(return_dict=True, **data)
            loss = output.loss
            accelerator.backward(loss)
            optim.step()
            scheduler.step()
            accelerator.log({"training_loss": loss}, step=index - 1)
            pbar.set_description(f"Loss: {loss}", refresh=True)

            gpus = GPUtil.getGPUs()
            for gpu_num in range(len(gpus)):
                gpu = gpus[gpu_num]
                if gpu.temperature >= TEMP_LIMIT:
                    faktor = int(gpu.temperature) - TEMP_LIMIT
                    time.sleep(faktor * 5)
