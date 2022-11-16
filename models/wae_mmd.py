import torch
from ml_collections import config_dict
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class WAE_MMD(BaseVAE):

    def __init__(self, config: config_dict.ConfigDict) -> None:
        super(WAE_MMD, self).__init__()

        # Build Encoder
        modules = []
        encoder_dims = [config.in_features] + list(config.hidden_dims)

        for i in range(len(config.hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=encoder_dims[i], out_features=encoder_dims[i + 1]),
                    nn.BatchNorm1d(encoder_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        self.encoder = nn.Sequential(*modules)

        self.fc_z = nn.Linear(config.hidden_dims[-1], config.latent_dim)
        # Build Decoder
        modules = []
        decoder_dims = [config.latent_dim] + list(reversed(config.hidden_dims)) + [config.in_features]

        for i in range(len(config.hidden_dims) + 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(decoder_dims[i], decoder_dims[i + 1]),
                    nn.BatchNorm1d(decoder_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Linear(config.in_features, config.in_features)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        z = self.fc_z(result)

        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the sample matrix space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C]
        """
        result = self.decoder(z)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = self.encode(input)
        return [self.decode(z), input, z]

    def loss_function(self,
                      *args,
                      config: config_dict.ConfigDict) -> dict:
        recons = args[0]
        input = args[1]
        z = args[2]

        recons_loss = F.mse_loss(recons, input)

        mmd_loss = self.compute_mmd(z, config)

        loss = recons_loss + mmd_loss
        return {'loss': loss, 'MSE': recons_loss, 'MMD': mmd_loss}

    def compute_mmd(self, z: Tensor, config: config_dict.ConfigDict) -> Tensor:

        prior_z = torch.randn_like(z)

        zz, zp, pp = torch.mm(z, z.t()), torch.mm(z, prior_z.t()), torch.mm(prior_z, prior_z.t())

        rzz = (zz.diag().unsqueeze(0).expand_as(zz))
        rpp = (pp.diag().unsqueeze(0).expand_as(pp))

        B = z.size(0)
        z_dim = z.size(1)
        if config.kernel_type == 'rbf':
            sigma = 1. / (2. * z_dim * config.latent_var)

            K = torch.exp(- sigma * (rzz.t() + rzz - 2 * zz))
            L = torch.exp(- sigma * (rpp.t() + rpp - 2 * pp))
            P = torch.exp(- sigma * (rzz.t() + rpp - 2 * zp))
        elif config.kernel_type == 'imq':
            # C value from WAE paper https://arxiv.org/pdf/1711.01558v1.pdf
            # expected squared distance between two multivariate Gaussian vectors drawn from prior_z
            C = 2. * z_dim * config.latent_var

            K = C / (1e-7 + C + rzz.t() + rzz - 2 * zz)
            L = C / (1e-7 + C + rpp.t() + rpp - 2 * pp)
            P = C / (1e-7 + C + rzz.t() + rpp - 2 * zp)

        beta = (1. / (B * (B - 1)))
        gamma = (2. / (B * B))
        mmd = beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P)

        return config.reg_weight * mmd
        # Sample from prior (Gaussian) distribution

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C]
        :return: (Tensor) [B x C]
        """
        return self.forward(x)[0]

    def latent(self, x: Tensor, **kwargs) -> Tensor:
        return self.encode(x)
