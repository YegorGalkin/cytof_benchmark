import torch
from ml_collections import config_dict
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class ReZero(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super(ReZero, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x) * self.alpha + x


class ResidualStack(torch.nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int) -> None:
        super(ResidualStack, self).__init__()
        self.hidden_dim = hidden_dim
        self.stack = nn.Sequential(*[ReZero(hidden_dim) for _ in range(n_layers)])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.stack(x)


class Encoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int) -> None:
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        layers = []
        if self.input_dim != self.hidden_dim:
            layers.append(nn.Linear(self.input_dim, self.hidden_dim))

        layers.append(ResidualStack(hidden_dim=self.hidden_dim, n_layers=self.n_layers))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


class Decoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int) -> None:
        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        layers = []
        if self.input_dim != self.hidden_dim:
            layers.append(nn.Linear(self.input_dim, self.hidden_dim))

        layers.append(ResidualStack(hidden_dim=self.hidden_dim, n_layers=self.n_layers))

        if self.output_dim != self.hidden_dim:
            layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


class CodeLayer(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 embed_dim: int,
                 embed_entries: int,
                 temperature: float,
                 kld_scale: float,
                 straight_through: bool) -> None:
        super(CodeLayer, self).__init__()

        self.in_features = in_features
        self.embed_dim = embed_dim
        self.embed_entries = embed_entries
        self.straight_through = straight_through
        self.temperature = temperature
        self.kld_scale = kld_scale

        if self.in_features != self.embed_dim:
            self.linear_in = nn.Linear(self.in_features, self.embed_entries)
        else:
            self.linear_in = nn.Identity()

        embed = torch.randn(self.embed_entries, self.embed_dim, dtype=torch.float32)
        self.register_buffer("embed", embed)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, float, torch.LongTensor]:
        x = self.linear_in(x.float())

        hard = self.straight_through if self.training else True

        soft_one_hot = F.gumbel_softmax(x, tau=self.temperature, dim=1, hard=hard)
        quantize = soft_one_hot @ self.embed

        # + kl divergence to the prior loss
        qy = F.softmax(x, dim=1)
        diff = torch.sum(qy * torch.log(qy * self.embed_entries + 1e-10), dim=1).mean()

        embed_ind = soft_one_hot.argmax(dim=1)

        return quantize, diff, embed_ind


class VQVAE(BaseVAE):

    def __init__(self, config: config_dict.ConfigDict) -> None:
        super(VQVAE, self).__init__()

        self.encoder = Encoder(input_dim=config.in_features, hidden_dim=config.hidden_features,
                               n_layers=config.n_layers)
        self.codebook1 = CodeLayer(in_features=config.hidden_features,
                                   embed_dim=config.embed_dim1,
                                   embed_entries=config.embed_entries1,
                                   temperature=config.temperature,
                                   kld_scale=config.kld_scale,
                                   straight_through=config.straight_through)
        self.codebook2 = CodeLayer(in_features=config.hidden_features + config.in_features,
                                   embed_dim=config.embed_dim2,
                                   embed_entries=config.embed_entries2,
                                   temperature=config.temperature,
                                   kld_scale=config.kld_scale,
                                   straight_through=config.straight_through)
        self.codebook3 = CodeLayer(in_features=config.hidden_features + config.in_features,
                                   embed_dim=config.embed_dim3,
                                   embed_entries=config.embed_entries3,
                                   temperature=config.temperature,
                                   kld_scale=config.kld_scale,
                                   straight_through=config.straight_through)

        self.decoder1 = Decoder(input_dim=config.embed_dim1,
                                hidden_dim=config.hidden_features,
                                output_dim=config.in_features,
                                n_layers=config.n_layers)

        self.decoder2 = Decoder(input_dim=config.embed_dim2 + config.embed_dim1,
                                hidden_dim=config.hidden_features,
                                output_dim=config.in_features,
                                n_layers=config.n_layers)

        self.decoder3 = Decoder(input_dim=config.embed_dim3 + config.embed_dim2,
                                hidden_dim=config.hidden_features,
                                output_dim=config.in_features,
                                n_layers=config.n_layers)
        self.kld_scale = config.kld_scale

    def forward(self, x):
        encoder_output = self.encoder(x)
        code_q1, code_d1, emb_id1 = self.codebook1(encoder_output)
        decoder_output1 = self.decoder1(code_q1)

        code_q2, code_d2, emb_id2 = self.codebook2(torch.cat([encoder_output, decoder_output1], axis=1))
        decoder_output2 = self.decoder2(torch.cat([code_q2, code_q1], axis=1))

        code_q3, code_d3, emb_id3 = self.codebook3(torch.cat([encoder_output, decoder_output2], axis=1))
        decoder_output3 = self.decoder3(torch.cat([code_q3, code_q2], axis=1))

        return decoder_output3, x, [code_d1, code_d2, code_d3], encoder_output, [emb_id1, emb_id2, emb_id3]

    def loss_function(self, *args) -> dict:
        recons = args[0]
        input = args[1]
        l_loss = sum(args[2])
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
