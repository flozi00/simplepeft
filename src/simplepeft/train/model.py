from torch.optim.lr_scheduler import ExponentialLR
import lightning.pytorch as pl
import GPUtil
import time


class lightningmodel(pl.LightningModule):
    def __init__(self, model_name, model, processor, optim, lr=1e-5):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.processor = processor
        self.lr = lr
        self.trained = 0
        self.optim = optim

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.trained += 1

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
            if gpu.temperature >= 74:
                time.sleep(4)

        self.log("train/loss", loss, prog_bar=True, on_step=True)

        # save model every 1000 steps, ignoring the gradient accumulation steps
        if self.trained % 1000 == 0 and self.trained != 0:
            try:
                self.model.push_to_hub(self.model_name)
                self.processor.push_to_hub(self.model_name)
            except Exception as e:
                print(e)
            self.model.save_pretrained(self.model_name)
            self.processor.save_pretrained(self.model_name)

        return loss

    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), lr=self.lr)
        scheduler = ExponentialLR(
            optimizer=optimizer,
            gamma=0.99999,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
