from pickletools import optimize
from pyexpat import features
import pytorch_lightning as pl
from typing import Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import spectral_norm
from utils import kaiming_init, load_checkpoint
from efficient_net import build_efficientnet_lite
from generator import Generator
from differentiable_augmentation import DiffAugment
from dataset import load_data

class DownBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, 4, 2, 1)
        self.bn = nn.BatchNorm2d(c_out)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.leaky_relu(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, channels, l):
        super(MultiScaleDiscriminator, self).__init__()
        self.head_conv = spectral_norm(nn.Conv2d(512, 1, 3, 1, 1))
        layers = [DownBlock(channels, 64 * [1, 2, 4, 8][l - 1])] + [DownBlock(64 * i, 64 * i * 2) for i in [1, 2, 4][l - 1:]]
        self.model = nn.Sequential(*layers)
        # self.optim = Adam(self.model.parameters(), lr=0.0002, betas=(0, 0.99))

    def forward(self, x):
        x = self.model(x)
        return self.head_conv(x)


class CSM(nn.Module):
    """
    Implementation for the proposed Cross-Scale Mixing.
    """

    def __init__(self, channels, conv3_out_channels):
        super(CSM, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels, conv3_out_channels, 3, 1, 1)

        for param in self.conv1.parameters():
            param.requires_grad = False

        for param in self.conv3.parameters():
            param.requires_grad = False

        self.apply(kaiming_init)

    def forward(self, high_res, low_res=None):
        batch, channels, width, height = high_res.size()
        if low_res is None:
            # high_res_flatten = rearrange(high_res, "b c h w -> b c (h w)")
            high_res_flatten = high_res.view(batch, channels, width * height)
            high_res = self.conv1(high_res_flatten)
            high_res = high_res.view(batch, channels, width, height)
            high_res = self.conv3(high_res)
            high_res = F.interpolate(high_res, scale_factor=2., mode="bilinear")
            return high_res
        else:
            high_res_flatten = high_res.view(batch, channels, width * height)
            high_res = self.conv1(high_res_flatten)
            high_res = high_res.view(batch, channels, width, height)
            high_res = torch.add(high_res, low_res)
            high_res = self.conv3(high_res)
            high_res = F.interpolate(high_res, scale_factor=2., mode="bilinear")
            return high_res


class litProjectedGAN(pl.LightningModule):
    def __init__(self, *args):
        super(litProjectedGAN, self).__init__()
        self.img_size = 100#args.image_size

        self.gen = Generator()#args.image_size)

        self.efficient_net = build_efficientnet_lite("efficientnet_lite1", 1000)
        self.efficient_net = nn.DataParallel(self.efficient_net)
        # checkpoint = torch.load("./efficientnet_lite0.pth")
        # load_checkpoint(self.efficient_net, checkpoint)
        self.efficient_net.eval()

        feature_sizes = self.get_feature_channels()
        self.csms = nn.ModuleList([
            CSM(feature_sizes[3], feature_sizes[2]),
            CSM(feature_sizes[2], feature_sizes[1]),
            CSM(feature_sizes[1], feature_sizes[0]),
            CSM(feature_sizes[0], feature_sizes[0]),
        ])

        self.discs = nn.ModuleList([
           MultiScaleDiscriminator(feature_sizes[0], 1),
           MultiScaleDiscriminator(feature_sizes[1], 2),
           MultiScaleDiscriminator(feature_sizes[2], 3),
           MultiScaleDiscriminator(feature_sizes[3], 4),
        ][::-1])

        self.latent_dim =100# args.latent_dim
        self.epochs = 2#args.epochs

        augmentations = 'color,translation,cutout'
        self.DiffAug = DiffAugment(augmentations)
        self.diff_aug = 'abc'#args.diff_aug

        # self.dataset = load_data(args.dataset_path, args.batch_size)
        self.log_every = 1#args.log_every
        # self.ckpt_path = args.checkpoint_path
        # self.save_all = args.save_all
        self.automatic_optimization = False

    
    def get_feature_channels(self):
        sample = torch.randn(1, 3, self.img_size, self.img_size)
        _, features = self.efficient_net(sample)
        return [f.shape[1] for f in features]

    def csm_forward(self, features):
        features = features[::-1]
        csm_features = []
        for i, csm in enumerate(self.csms):
            if i == 0:
                d = csm(features[i])
                csm_features.append(d)
            else:
                d = csm(features[i], d)
                csm_features.append(d)
        return csm_features #features

    def csm_forward_idx(self, features, idx):
        features = features[::-1]
        csm_features = []
        if idx == 0:
            d = self.csms[idx](features[idx])
            csm_features.append(d)
        else:
            for i, csm in enumerate(self.csms[:idx+1]):
                if i == 0:
                    d = csm(features[i])
                    csm_features.append(d)
                else:
                    d = csm(features[i], d)
                    csm_features.append(d)
        return csm_features

    # used at inferece time
    def forward(self, x): # x is a batch of z vectors
        return self.gen(x)

    def training_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx, optimizer_idx):
        z = torch.randn(batch.shape[0], self.latent_dim, device=self.device)
        gen_imgs_disc = self.gen(z).detach()
        if self.diff_aug:
            gen_imgs_disc = self.DiffAug(gen_imgs_disc)
            real_imgs = self.DiffAug(batch)
        
        if optimizer_idx != 0:
            _, feature_fake = self.efficient_net(gen_imgs_disc)
            _, feature_real = self.efficient_net(real_imgs)
            feature_real = self.csm_forward_idx(feature_real, optimizer_idx-1)
            features_fake = self.csm_forward_idx(feature_fake, optimizer_idx-1)
            disc_losses = []
            # for feature_real, feature_fake, disc in zip(features_real, features_fake, self.discs):
            y_hat_real = self.discs[optimizer_idx-1](feature_real)  # Cx4x4
            y_hat_fake = self.discs[optimizer_idx-1](feature_fake)  # Cx4x4
            y_hat_real = y_hat_real.sum(1)  # sum along channels axis (is 1 anyways, however it still removes the unnecessary axis)
            y_hat_fake = y_hat_fake.sum(1)
            loss_real = torch.mean(F.relu(1. - y_hat_real))
            self.log(f"disc_lossReal_{optimizer_idx-1}", loss_real)
            loss_fake = torch.mean(F.relu(1. + y_hat_fake))
            self.log(f"disc_lossFake_{optimizer_idx-1}", loss_fake)
            disc_loss = loss_real + loss_fake
            self.log(f"disc_loss_{optimizer_idx}", disc_loss)
            return {"loss": disc_losses}

        # Train Generator:
        if optimizer_idx == 0:
            z = torch.randn(real_imgs.shape[0], self.latent_dim, device=self.device)
            gen_imgs_gen = self.gen(z)

            if self.diff_aug:
                gen_imgs_gen = self.DiffAug.forward(gen_imgs_gen)

            # get efficient net features
            _, features_fake = self.efficient_net(gen_imgs_gen)

            # feed efficient net features through CSM
            features_fake = self.csm_forward(features_fake)

            gen_loss = 0.
            for feature_fake, disc in zip(features_fake, self.discs):
                y_hat = disc(feature_fake)
                y_hat = y_hat.sum(1)
                gen_loss = -torch.mean(y_hat)
            self.log("gen_loss", gen_loss)
            return {"loss":gen_loss}

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)
    
    def configure_optimizers(self):
        optimizers_list = [Adam(self.gen.parameters(), lr=self.args.lr, betas=(self.args.beta1,self.args.beta2))]
        optimizers_list += [Adam(disc.parameters(),lr=0.0002, betas=(0,0.99)) for disc in self.discs]
        return [optimizers_list], [] # empty lr-scheduler list. Dunno what to use.