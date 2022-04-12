from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from lit_projectedGAN import litProjectedGAN
from dataset import load_data
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ProjectedGAN")
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR', help='learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.0, metavar='lambda', help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='lambda', help='Adam beta param (default: 0.999)')
    parser.add_argument('--latent-dim', type=int, default=100, help='Latent dimension for generator (default: 100)')
    parser.add_argument('--diff-aug', type=bool, default=True, help='Apply differentiable augmentation to both discriminator and generator (default: True)')
    parser.add_argument('--checkpoint-path', type=str, default="/checkpoints", metavar='Path', help='Path for checkpointing (default: /checkpoints)')
    parser.add_argument('--save-all', type=bool, default=False, help='Saves all discriminator, all CSMs and generator if True, only the generator otherwise (default: False)')
    parser.add_argument('--checkpoint-efficient-net', type=str, default="efficientnet_lite1.pth", metavar='Path', help='Path for EfficientNet checkpoint (default: efficientnet_lite1.pth)')
    parser.add_argument('--log-every', type=int, default=100, help='How often model will be saved, generated images will be saved etc. (default: 100)')
    parser.add_argument('--dataset-path', type=str, default='/data', metavar='Path', help='Path to data (default: /data)')
    parser.add_argument('--image-size', type=int, default=256, help='Size of images in dataset (default: 256)')
    args = parser.parse_args()
    model = litProjectedGAN(args)
    data = load_data(args.dataset_path)
    wandb_logger = WandbLogger()
    trainer = Trainer(
        devices=1, accelerator="gpu",
        log_every_n_steps=args.log_every, max_epochs=args.epochs,
        num_sanity_val_steps=0, logger=wandb_logger, precision=16
    )
    trainer.fit(model=model, train_dataloader=data)

    # self.ckpt_path = args.checkpoint_path