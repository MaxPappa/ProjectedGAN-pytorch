from dataclasses import dataclass

@dataclass
class configParams:
    batch_size: int = 8
    epochs: int = 100
    lr: float = 0.0002
    beta1: float = 0.0
    beta2: float = 0.999
    latent_dim: int = 100
    diff_aug: bool = True
    checkpoint_path: str = "./checkpoints"
    checkpoint_efficient_net: str = "efficientnet_lite1.pth"
    log_every: int = 10
    dataset_path: str = "./data2"
    image_size: int = 256