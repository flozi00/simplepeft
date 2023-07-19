from lion_pytorch import Lion
from accelerate import Accelerator
import warnings
from tqdm.auto import tqdm
import time
import GPUtil
from torch.optim.lr_scheduler import ExponentialLR

warnings.simplefilter("ignore")

accelerator = Accelerator(gradient_accumulation_steps=4, log_with="wandb")
device = accelerator.device

accelerator.init_trackers("huggingface")


def start_training(
    model, processor, dloader, PEFT_MODEL, LR: float, model_conf: dict, batch_size: int
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
    if model_conf["is8bit"]:
        from bitsandbytes.optim import PagedLion

        optim = PagedLion(model.parameters(), lr=LR)
    else:
        optim = Lion(model.parameters(), lr=LR)

    scheduler = ExponentialLR(optim, gamma=0.9)
    model, optim, dloader, scheduler = accelerator.prepare(
        model, optim, dloader, scheduler
    )

    model.train()
    index = 0
    for data in tqdm(dloader):
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
            accelerator.log({"training_loss": loss}, step=index)

            for xyz in range(10):
                gpus = GPUtil.getGPUs()
                for gpu_num in range(len(gpus)):
                    gpu = gpus[gpu_num]
                    if gpu.temperature >= 68:
                        time.sleep(3)
