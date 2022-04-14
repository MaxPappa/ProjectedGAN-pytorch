import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from lit_projectedGAN import litProjectedGAN
from dataset import load_data
import config
from os import mkdir
import wandb

class EpochModelCheckpoint(pl.Callback):
    def __init__(self, num_epoch: int):
        self.num_epoch = num_epoch

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (trainer.current_epoch % self.num_epoch) == 0:
            trainer.save_checkpoint(f"./checkpoints/epoch={trainer.current_epoch}_LitProjectedGANArt.pth")



if __name__ == '__main__':
    mkdir("./checkpoints")
    cfg = config.configParams()
    model = litProjectedGAN(cfg)
    data = load_data(cfg.dataset_path, cfg.batch_size)
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(
        devices=1, accelerator="gpu",
        log_every_n_steps=cfg.log_every, max_epochs=cfg.epochs,
        num_sanity_val_steps=0, logger=wandb_logger,# precision=16,
        callbacks=[EpochModelCheckpoint(num_epoch=cfg.num_epoch_checkpoint)]
    )
    trainer.fit(model, data)
    wandb.save(f"./checkpoints/*")
    wandb.finish()