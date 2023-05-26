from torch.optim.lr_scheduler import ExponentialLR
import lightning.pytorch as pl
import GPUtil
import time


class lightningmodel(pl.LightningModule):
    def __init__(
        self, model_name, model, processor, optim, lr=1e-5, save_every_hours=6
    ):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.processor = processor
        self.lr = lr
        self.optim = optim
        self.start_time = time.time()
        self.save_every_hours = save_every_hours
        self.processor.save_pretrained(self.model_name)
        self.processor.push_to_hub(self.model_name)

    def forward(self, **inputs):
        return self.model(return_dict=True, **inputs)

    def training_step(self, batch, batch_idx):
        elapsed_time = time.time() - self.start_time
        outputs = self(**batch)
        loss = outputs.loss

        # iterate over all gpus and log temperature and load, if temperature is above 74, wait 4 seconds to cool down
        gpus = GPUtil.getGPUs()
        for gpu_num in range(len(gpus)):
            gpu = gpus[gpu_num]
            self.log(
                f"train/gpu_temp_{gpu_num}",
                gpu.temperature,
                prog_bar=True,
                on_step=True,
            )
            self.log(
                f"train/gpu_load_{gpu_num}", gpu.memoryUtil, prog_bar=True, on_step=True
            )
            if gpu.temperature >= 72:
                time.sleep(10)

        self.log("train/loss", loss, prog_bar=True, on_step=True)

        if batch_idx % (100 * self.save_every_hours) == 0 and batch_idx != 0:
            self.model.save_pretrained(self.model_name)

        # push to hub every X hours
        if elapsed_time > (
            60 * 60 * self.save_every_hours
        ):  # 6 hours (60 seconds * 60 minutes * hours)
            self.start_time = time.time()
            try:
                self.model.push_to_hub(self.model_name)
            except Exception as e:
                print(e)

        # this is an custom learning rate sheduler
        # while the learning rate is above 5e-6, the formula is 2 divided by the loss
        # the lower the loss, the more often the learning rate is decreased
        # this allows for a faster training at the beginning because
        # we have a higher learning rate and a lower learning rate at the end
        # when the learning rate is below 5e-6, the learning rate is decreased every x steps
        # this time x is 1 divided by the loss, so the lower the loss the less often the learning rate is decreased
        # notice: while experiments on whisper fine-tuning, the loss reached 0.2 after 30 steps and 0.1 after 60 steps
        if loss > 0:
            if self.optimizers().param_groups[0]["lr"] >= 1e-5:
                for s in range(int(2 / loss)):
                    self.lr_schedulers().step()
            else:
                if batch_idx % int(1 / loss) == 0:
                    self.lr_schedulers().step()

        return loss

    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), lr=self.lr)
        scheduler = ExponentialLR(
            optimizer=optimizer,
            gamma=0.999,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
