from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from lit_projectedGAN import litProjectedGAN
from dataset import load_data
import config

if __name__ == '__main__':
    cfg = config.configParams()
    model = litProjectedGAN(cfg)
    data = load_data(cfg.dataset_path, cfg.batch_size)
    wandb_logger = WandbLogger()
    trainer = Trainer(
        devices=1, accelerator="gpu",
        log_every_n_steps=cfg.log_every, max_epochs=cfg.epochs,
        num_sanity_val_steps=0, logger=wandb_logger#, precision=16
    )
    trainer.fit(model, data)