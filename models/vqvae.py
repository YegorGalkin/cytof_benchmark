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
            self.linear_in = nn.Linear(config.hidden_features, config.embed_dim)
        else:
            self.linear_in = nn.Identity()

        self.dim = config.embed_dim
        self.n_embed = config.nb_entries
        self.decay = config.decay
        self.eps = config.eps

        embed = torch.randn(self.dim, self.n_embed, dtype=torch.float32)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(self.n_embed, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, float, torch.LongTensor]:
        x = self.linear_in(x.float())

        dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * x @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(x.dtype)

        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = x.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id: torch.LongTensor) -> torch.FloatTensor:
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class VQVAE(BaseVAE):

    def __init__(self, config: config_dict.ConfigDict) -> None:
        super(VQVAE, self).__init__()

        self.encoder = Encoder(config)
        self.codebook = Decoder(config)
        self.decoder = CodeLayer(config)

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
        loss = r_loss + self.beta * l_loss
        return {'loss': loss, 'MSE': r_loss.detach(), 'Qloss': l_loss.detach()}

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input sample matrix x, returns the reconstructed sample matrix
        :param x: (Tensor) [B x C]
        :return: (Tensor) [B x C]
        """

        return self.forward(x)[0]
