import torch
from ml_collections import config_dict
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class BetaVAE(BaseVAE):

    def __init__(self, config: config_dict.ConfigDict) -> None:
        super(BetaVAE, self).__init__()

        self.kld_weight = torch.Tensor([config.kld_weight]).to(config.device)

        self.loss_type = config.loss_type

        self.act_class = getattr(nn, config.activation)

        if self.loss_type == 'disentangled_beta':
            self.C_max = torch.Tensor([config.C_max]).to(config.device)
            self.C_stop_iter = torch.Tensor([config.C_stop_iter]).to(config.device)

        modules = []

        # Build Encoder

        encoder_dims = [config.in_features] + list(config.hidden_dims)

        for i in range(len(config.hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=encoder_dims[i], out_features=encoder_dims[i + 1]),
                    nn.BatchNorm1d(encoder_dims[i + 1]),
                    self.act_class(),
                )
            )
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(config.hidden_dims[-1], config.latent_dim)
        self.fc_var = nn.Linear(config.hidden_dims[-1], config.latent_dim)

        # Build Decoder
        modules = []
        decoder_dims = [config.latent_dim] + list(reversed(config.hidden_dims)) + [config.in_features]

        for i in range(len(config.hidden_dims) + 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(decoder_dims[i], decoder_dims[i + 1]),
                    nn.BatchNorm1d(decoder_dims[i + 1]),
                    self.act_class()
                )
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
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the sample matrix space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C]
        """
        # Only batch - normalized layers
        result = self.decoder(z)
        # Use linear layer to map normalized decoder outputs back to input space
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      epoch: int = 0) -> dict:
        r"""
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param epoch: current epoch
        :param args:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'beta':
            loss = recons_loss + self.kld_weight * kld_loss
        elif self.loss_type == 'disentangled_beta':
            C = torch.clamp(self.C_max / self.C_stop_iter * epoch, 0, self.C_max.data[0])
            loss = recons_loss + self.kld_weight * (kld_loss - C).abs()
        else:
            return {}

        return {'loss': loss, 'MSE': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input sample matrix x, returns the reconstructed sample matrix
        :param x: (Tensor) [B x C]
        :return: (Tensor) [B x C]
        """

        return self.forward(x)[0]

    def latent(self, x: Tensor, **kwargs) -> Tensor:
        return self.reparameterize(*self.encode(x))
