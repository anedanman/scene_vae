import pytorch_lightning as pl
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
from utils import get_rot_matr
import torchvision.models as models
from model import Encoder, Decoder
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as F


class PlSceneVAE(pl.LightningModule):
    def __init__(self, encoder_dim=1024, latent_dim=1024, beta=1):
        super().__init__()
        self.beta = beta
        self.encoder = Encoder()
        self.decoder = nn.Sequential(nn.Linear(latent_dim, encoder_dim), nn.GELU(), Decoder())
        self.fc_mu = nn.Linear(encoder_dim, latent_dim)
        self.fc_var = nn.Linear(encoder_dim, latent_dim)
        self.feed_fwd = nn.Sequential(nn.Linear(latent_dim, latent_dim * 2), nn.GELU(), nn.Linear(latent_dim * 2, latent_dim))
        self.activation = nn.GELU()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, inputs, labels):
        encoded_inputs = self.encoder(inputs)
        mu = self.fc_mu(encoded_inputs)
        logvar = self.fc_var(encoded_inputs)
        samples = self.reparameterize(mu, logvar)
        z_i = self.activation(self.feed_fwd(samples))
        
        z_i = z_i.transpose(0, 1)
        mask = labels.unsqueeze(-1).expand(z_i.size())
        z_i *= mask
        z_i = z_i.transpose(0, 1)
        z = torch.sum(z_i, axis=0)

        mus = (mu.transpose(0, 1) * mask).transpose(0, 1)
        mus = torch.mean(mus, axis=0)
        logvars = (logvar.transpose(0, 1) * mask).transpose(0, 1)
        logvars = torch.mean(logvars, axis=0)
        
        return self.decoder(z), mus, logvars, z_i

    def training_step(self, batch, batch_idx):
        scene = batch['scene']
        masks = batch['masks']
        labels = batch['labels']
        rec, mus, logvars, _ = self(masks, labels)
        loss = self.loss_func(rec, scene, mus, logvars)
        return loss[0]

    def validation_step(self, batch, batch_idx):
        scene = batch['scene']
        masks = batch['masks']
        labels = batch['labels']
        rec, mus, logvars, _ = self(masks, labels)
        loss = self.loss_func(rec, scene, mus, logvars)
        return loss[0]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss_func(self, recon_x, x, mus, logvars):
        mse_loss = torch.nn.MSELoss(reduction='sum')
        mse = mse_loss(recon_x, x)
        kld = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
        return mse + kld, mse, kld
