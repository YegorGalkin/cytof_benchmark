import torch
import torch.utils.data
from ml_collections import config_dict
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

from models.hyperspherical_vae_extra.distributions import VonMisesFisher
from models.hyperspherical_vae_extra.distributions import HypersphericalUniform


class HyperSphericalVAE(BaseVAE):

    def __init__(self, config: config_dict.ConfigDict) -> None:
        super(HyperSphericalVAE, self).__init__()

        self.kld_weight = torch.Tensor([config.kld_weight]).to(config.device)
        self.z_dim = config.latent_dim
        self.device = config.device
        self.act_class = getattr(nn, config.activation)

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
        self.fc_var = nn.Linear(config.hidden_dims[-1], 1)

        # Build Decoder
        modules = []
        decoder_dims = [config.latent_dim] + list(reversed(config.hidden_dims)) + [config.in_features]

        for i in range(len(config.hidden_dims) + 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(decoder_dims[i], decoder_dims[i + 1]),
                    nn.BatchNorm1d(decoder_dims[i + 1]),
                    self.act_class())
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

        # Split the result into mean and concentration components
        # of the latent von Mises-Fisher distribution
        z_mean = self.fc_mu(result)
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)

        z_var = F.softplus(self.fc_var(result)) + 1

        return [z_mean, z_var]

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

    def reparameterize(self, z_mean: Tensor, z_var: Tensor) -> Tensor:
        """

        :return: (Tensor) [B x D]
        """
        q_z = VonMisesFisher(z_mean, z_var)
        p_z = HypersphericalUniform(self.z_dim - 1, device=self.device)

        return q_z, p_z

    def forward(self, x: Tensor, **kwargs) -> list[tuple[Tensor, Tensor] | Tensor]:

        z_mean, z_var = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        x_ = self.decode(z)

        return [x_, x, q_z, p_z]

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
        q_z = args[2]
        p_z = args[3]

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

        loss = recons_loss + self.kld_weight * kld_loss

        return {'loss': loss, 'MSE': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input sample matrix x, returns the reconstructed sample matrix
        :param x: (Tensor) [B x C]
        :return: (Tensor) [B x C]
        """

        return self.forward(x)[0]

    def latent(self, x: Tensor, **kwargs) -> Tensor:
        return self.reparameterize(*self.encode(x))[0].rsample()
