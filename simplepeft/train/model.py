from torch.optim.lr_scheduler import ExponentialLR
import lightning.pytorch as pl
import time
import GPUtil


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

        self.log("train/loss", loss, prog_bar=True, on_step=True)

        for xyz in range(5):
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
                    f"train/gpu_load_{gpu_num}",
                    gpu.memoryUtil,
                    prog_bar=True,
                    on_step=True,
                )
                if gpu.temperature >= 68:
                    time.sleep(10)

        if batch_idx % 250 == 0 and batch_idx != 0:
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

        if batch_idx >= 500:
            self.lr_schedulers().step()

        return loss

    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), lr=self.lr)
        scheduler = ExponentialLR(
            optimizer=optimizer,
            gamma=0.9999,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
