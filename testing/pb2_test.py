from typing import List

import numpy as np
import ray
import torch
import torch.optim as optim
from ml_collections import config_dict
from ray import tune, air
from ray.air import session, Checkpoint
from ray.tune.schedulers.pb2 import PB2
from torch import nn
from torch.nn import functional as F

from datasets import OrganoidDatasetDeprecated


def get_config():
    config = config_dict.ConfigDict()
    # General parameters
    config.dataset = 'Organoid'
    config.model = 'VAE'
    config.seed = 12345
    config.output_dir = './logs/VanillaVAE/'
    config.device = 'cuda'
    config.epochs = 10000

    # VAE architecture parameters
    config.architecture = config_dict.ConfigDict()
    config.architecture.in_features = 41
    config.architecture.latent_dim = 2
    config.architecture.hidden_dims = [256, 256, 256, 256, 256]
    config.architecture.kld_weight = 0.0025
    config.architecture.loss_type = 'beta'
    config.architecture.activation = 'LeakyReLU'

    # Tunable default parameters
    config.tunable = config_dict.ConfigDict()
    config.tunable.learning_rate = 0.05
    config.tunable.weight_decay = 0.0
    config.tunable.batch_size = 4096
    return config


class BetaVAE(nn.Module):

    def __init__(self, config: config_dict.ConfigDict) -> None:
        super(BetaVAE, self).__init__()

        self.config = config
        self.kld_weight = torch.Tensor([config.architecture.kld_weight]).to(config.device)

        self.act_class = getattr(nn, config.architecture.activation)

        modules = []

        # Build Encoder

        encoder_dims = [config.architecture.in_features] + list(config.architecture.hidden_dims)

        for i in range(len(config.architecture.hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=encoder_dims[i], out_features=encoder_dims[i + 1]),
                    nn.BatchNorm1d(encoder_dims[i + 1]),
                    self.act_class(),
                )
            )
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(config.architecture.hidden_dims[-1], config.architecture.latent_dim)
        self.fc_var = nn.Linear(config.architecture.hidden_dims[-1], config.architecture.latent_dim)

        # Build Decoder
        modules = []
        decoder_dims = \
            [config.architecture.latent_dim] + \
            list(reversed(config.architecture.hidden_dims)) + \
            [config.architecture.in_features]

        for i in range(len(config.architecture.hidden_dims) + 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(decoder_dims[i], decoder_dims[i + 1]),
                    nn.BatchNorm1d(decoder_dims[i + 1]),
                    self.act_class()
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Linear(config.architecture.in_features, config.architecture.in_features)

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
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

    def decode(self, z: torch.Tensor) -> torch.Tensor:
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

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
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

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args) -> dict:
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

        loss = recons_loss + self.kld_weight * kld_loss

        return {'loss': loss, 'MSE': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an input sample matrix x, returns the reconstructed sample matrix
        :param x: (Tensor) [B x C]
        :return: (Tensor) [B x C]
        """

        return self.forward(x)[0]

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.reparameterize(*self.encode(x))


def train(model, optimizer, train_dataloader):
    for X_batch in train_dataloader:
        optimizer.zero_grad()
        model.train()
        outputs = model.forward(X_batch)
        loss = model.loss_function(*outputs)

        loss['loss'].backward()
        optimizer.step()


def test(model, val_dataloader):
    with torch.no_grad():
        model.eval()
        mse, kld, loss = list(), list(), list()
        for X_batch in val_dataloader:
            loss_dict = dict()
            outputs = model.forward(X_batch)
            losses = model.loss_function(*outputs)
            for key in losses.keys():
                loss_dict[key] = losses[key].item() * X_batch.shape[0]
            mse.append(loss_dict['MSE'])
            kld.append(loss_dict['KLD'])
            loss.append(loss_dict['loss'])
    data_len = sum(len(batch) for batch in val_dataloader)

    return \
        np.nan_to_num(sum(mse) / data_len, nan=10), \
        np.nan_to_num(sum(kld) / data_len, nan=10), \
        np.nan_to_num(sum(loss) / data_len, nan=10)


def vae_train(cfg):
    config = cfg.get('default_config')
    model = BetaVAE(config).to(config.device)

    optimizer = optim.Adam(model.parameters(),
                           lr=10**cfg.get("log10_lr"),
                           )

    dataset = OrganoidDatasetDeprecated(data_dir='/data/organoids')
    X_train, y_train = dataset.train
    X_val, y_val = dataset.val
    X_train_batches = torch.split(X_train, split_size_or_sections=int(2**cfg.get("log2_batch")))
    X_val_batches = torch.split(X_val, split_size_or_sections=int(2**cfg.get("log2_batch")))

    # Remove last batch if it has less than half batch size samples to reduce variance
    if X_train_batches[-1].shape[0] < int(2**cfg.get("log2_batch")) // 2:
        X_train_batches = X_train_batches[:-1]

    step = 1
    if session.get_checkpoint():
        checkpoint_dict = session.get_checkpoint().to_dict()

        model.load_state_dict(checkpoint_dict["model"])
        optimizer.load_state_dict(checkpoint_dict["optim"])
        # Note: Make sure to increment the loaded step by 1 to get the
        # current step.
        last_step = checkpoint_dict["step"]
        step = last_step + 1

        # NOTE: It's important to set the optimizer learning rates
        # again, since we want to explore the parameters passed in by PBT.
        # Without this, we would continue using the exact same
        # configuration as the trial whose checkpoint we are exploiting.
        if "learning_rate" in cfg:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 10**cfg.get("log10_lr")
    while True:
        train(model, optimizer, X_train_batches)
        MSE, KLD, loss = test(model, X_val_batches)

        checkpoint = None
        if step % cfg["checkpoint_interval"] == 0:
            checkpoint = Checkpoint.from_dict(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "step": step,
                }
            )
        session.report(
            {
                "MSE": MSE,
                "KLD": KLD,
                "loss": loss,
                'lr': 10**cfg.get("log10_lr"),
                "step": step,
            },
            checkpoint=checkpoint,
        )
        step += 1


if __name__ == '__main__':
    ray.init()

    perturbation_interval = 25
    scheduler = PB2(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        hyperparam_bounds={
            "log10_lr": [-10, 5],
            "log2_batch": [10, 15],
        },
    )

    tuner = tune.Tuner(
        tune.with_resources(vae_train, {"cpu": 8, "gpu": 0.25}),
        run_config=air.RunConfig(
            name="vae_training",
            verbose=2,
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=8,
            time_budget_s=60*60*4,
            scheduler=scheduler,
        ),
        param_space={
            # Define how initial values of the learning rates should be chosen.
            "log10_lr": tune.uniform(-5, -1),
            "log2_batch": tune.uniform(10, 15),
            "checkpoint_interval": perturbation_interval,
            "default_config": get_config()
        },
    )
    results_grid = tuner.fit()

    result_dfs = [result.metrics_dataframe for result in results_grid]
    best_result = results_grid.get_best_result(metric="MSE", mode="min")

    torch.save(best_result.checkpoint.to_dict(), '/logs/ray_tune/Beta_VAE_big_PB2.pth')
    with open('/logs/ray_tune/Beta_VAE_big_summary_PB2.txt', 'w') as f:
        print('val_mse,{}'.format(best_result.metrics['MSE']), file=f)
        print('val_kld,{}'.format(best_result.metrics['KLD']), file=f)
        print('val_loss,{}'.format(best_result.metrics['loss']), file=f)
