import torch
from ml_collections import config_dict
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class ReZero(torch.nn.Module):
    def __init__(self, config: config_dict.ConfigDict) -> None:
        super(ReZero, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(config.hidden_features, config.hidden_features),
            nn.BatchNorm1d(config.hidden_features),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x) * self.alpha + x


class ResidualStack(torch.nn.Module):
    def __init__(self, config: config_dict.ConfigDict) -> None:
        super(ResidualStack, self).__init__()

        self.stack = nn.Sequential(*[ReZero(config) for _ in range(config.n_layers)])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stack(x)


class Encoder(torch.nn.Module):
    def __init__(self, config: config_dict.ConfigDict) -> None:
        super(Encoder, self).__init__()

        layers = []
        if config.in_features != config.hidden_features:
            layers.append(nn.Linear(config.in_features, config.hidden_features))

        layers.append(ResidualStack(config))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


class Decoder(torch.nn.Module):
    def __init__(self, config: config_dict.ConfigDict) -> None:
        super(Decoder, self).__init__()

        layers = []
        if config.embed_dim != config.hidden_features:
            layers.append(nn.Linear(config.embed_dim, config.hidden_features))

        layers.append(ResidualStack(config))

        if config.in_features != config.hidden_features:
            layers.append(nn.Linear(config.hidden_features, config.in_features))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


class CodeLayer(torch.nn.Module):
    def __init__(self, config: config_dict.ConfigDict) -> None:
        super(CodeLayer, self).__init__()
        if config.hidden_features != config.embed_dim:
            self.linear_in = nn.Linear(config.hidden_features, config.nb_entries)
        else:
            self.linear_in = nn.Identity()

        self.embed_dim = config.embed_dim
        self.nb_entries = config.nb_entries
        self.straight_through = config.straight_through
        self.temperature = config.temperature
        self.kld_scale = config.kld_scale

        embed = torch.randn(self.nb_entries, self.embed_dim, dtype=torch.float32)
        self.register_buffer("embed", embed)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, float, torch.LongTensor]:
        x = self.linear_in(x.float())

        hard = self.straight_through if self.training else True

        soft_one_hot = F.gumbel_softmax(x, tau=self.temperature, dim=1, hard=hard)
        quantize = soft_one_hot @ self.embed

        # + kl divergence to the prior loss
        qy = F.softmax(x, dim=1)
        diff = torch.sum(qy * torch.log(qy * self.nb_entries + 1e-10), dim=1).mean()

        embed_ind = soft_one_hot.argmax(dim=1)

        return quantize, diff, embed_ind


class VQVAE(BaseVAE):

    def __init__(self, config: config_dict.ConfigDict) -> None:
        super(VQVAE, self).__init__()

        self.encoder = Encoder(config)
        self.codebook = CodeLayer(config)
        self.decoder = Decoder(config)

        self.kld_scale = config.kld_scale

    def forward(self, x):
        encoder_output = self.encoder(x)
        code_q, code_d, emb_id = self.codebook(encoder_output)
        decoder_output = self.decoder(code_q)

        return decoder_output, x, code_d, encoder_output, emb_id

    def loss_function(self, *args) -> dict:
        recons = args[0]
        input = args[1]
        l_loss = args[2]
        r_loss = recons.sub(input).pow(2).mean()
        loss = r_loss + self.kld_scale * l_loss
        return {'loss': loss, 'MSE': r_loss.detach(), 'KLD': l_loss.detach()}

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input sample matrix x, returns the reconstructed sample matrix
        :param x: (Tensor) [B x C]
        :return: (Tensor) [B x C]
        """

        return self.forward(x)[0]
