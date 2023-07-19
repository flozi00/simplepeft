from lion_pytorch import Lion
from accelerate import Accelerator
import warnings
from tqdm.auto import tqdm
import time
import GPUtil

warnings.simplefilter("ignore")

accelerator = Accelerator()
device = accelerator.device


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

    model, optim, dloader = accelerator.prepare(model, optim, dloader)

    model.train()
    for epoch in range(10):
        index = 0
        for data in tqdm(dloader):
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
            total_loss = loss.detach().float()
            print(total_loss)
            accelerator.backward(loss)
            optim.step()

            for xyz in range(10):
                gpus = GPUtil.getGPUs()
                for gpu_num in range(len(gpus)):
                    gpu = gpus[gpu_num]
                    if gpu.temperature >= 68:
                        time.sleep(3)
