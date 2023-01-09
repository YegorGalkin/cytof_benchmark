from typing import List

import ray
import torch
import torch.optim as optim
from ml_collections import config_dict
from ray import tune, air
from ray.air import session, Checkpoint
from ray.tune.examples.mnist_pytorch import ConvNet, get_data_loaders, train, test
from ray.tune.schedulers import PopulationBasedTraining
from torch import nn
from torch.nn import functional as F


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
    config.architecture.hidden_dims = (32, 32, 32)
    config.architecture.kld_weight = 0.0025
    config.architecture.loss_type = 'beta'
    config.architecture.activation = 'GELU'

    # Tunable parameters
    config.tunable = config_dict.ConfigDict()
    config.tunable.learning_rate = 0.05
    config.tunable.weight_decay = 0.0
    config.tunable.batch_size = 4096
    return config


class BetaVAE(nn.Module):

    def __init__(self, config: config_dict.ConfigDict) -> None:
        super(BetaVAE, self).__init__()

        self.kld_weight = torch.Tensor([config.kld_weight]).to(config.device)

        self.loss_type = config.loss_type

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

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
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


def train_convnet(config):
    # Create our data loaders, model, and optmizer.
    step = 1
    train_loader, test_loader = get_data_loaders()
    model = ConvNet()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("lr", 0.01),
        momentum=config.get("momentum", 0.9),
    )

    # If `session.get_checkpoint()` is not None, then we are resuming from a checkpoint.
    if session.get_checkpoint():
        # Load model state and iteration step from checkpoint.
        checkpoint_dict = session.get_checkpoint().to_dict()
        model.load_state_dict(checkpoint_dict["model_state_dict"])
        # Load optimizer state (needed since we're using momentum),
        # then set the `lr` and `momentum` according to the config.
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            if "lr" in config:
                param_group["lr"] = config["lr"]
            if "momentum" in config:
                param_group["momentum"] = config["momentum"]

        # Note: Make sure to increment the checkpointed step by 1 to get the current step.
        last_step = checkpoint_dict["step"]
        step = last_step + 1

    while True:
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)
        checkpoint = None
        if step % config["checkpoint_interval"] == 0:
            # Every `checkpoint_interval` steps, checkpoint our current state.
            checkpoint = Checkpoint.from_dict({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            })

        session.report(
            {"mean_accuracy": acc, "lr": config["lr"]},
            checkpoint=checkpoint
        )
        step += 1


perturbation_interval = 5
scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=perturbation_interval,
    metric="mean_accuracy",
    mode="max",
    hyperparam_mutations={
        # distribution for resampling
        "lr": tune.uniform(0.0001, 1),
        # allow perturbations within this set of categorical values
        "momentum": [0.8, 0.9, 0.99],
    },
)

if ray.is_initialized():
    ray.shutdown()
ray.init()

tuner = tune.Tuner(
    train_convnet,
    run_config=air.RunConfig(
        name="pbt_test",
        # Stop when we've reached a threshold accuracy, or a maximum
        # training_iteration, whichever comes first
        stop={"mean_accuracy": 0.96, "training_iteration": 50},
        verbose=1,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_score_attribute="mean_accuracy",
            num_to_keep=4,
        ),
        local_dir="/tmp/ray_results",
    ),
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=4,
    ),
    param_space={
        "lr": tune.uniform(0.001, 1),
        "momentum": tune.uniform(0.001, 1),
        "checkpoint_interval": perturbation_interval,
    },
)

results_grid = tuner.fit()

import matplotlib.pyplot as plt

# Get the best trial result
best_result = results_grid.get_best_result(metric="mean_accuracy", mode="max")

# Print `log_dir` where checkpoints are stored
print('Best result logdir:', best_result.log_dir)

# Print the best trial `config` reported at the last iteration
# NOTE: This config is just what the trial ended up with at the last iteration.
# See the next section for replaying the entire history of configs.
print('Best final iteration hyperparameter config:\n', best_result.config)

# Plot the learning curve for the best trial
df = best_result.metrics_dataframe
# Deduplicate, since PBT might introduce duplicate data
df = df.drop_duplicates(subset="training_iteration", keep="last")
df.plot("training_iteration", "mean_accuracy")
plt.xlabel("Training Iterations")
plt.ylabel("Test Accuracy")
plt.show()

def dcgan_train(config):
    use_cuda = config.get("use_gpu") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    netD = Discriminator().to(device)
    netD.apply(weights_init)
    netG = Generator().to(device)
    netG.apply(weights_init)
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(
        netD.parameters(), lr=config.get("lr", 0.01), betas=(beta1, 0.999)
    )
    optimizerG = optim.Adam(
        netG.parameters(), lr=config.get("lr", 0.01), betas=(beta1, 0.999)
    )
    with FileLock(os.path.expanduser("~/ray_results/.data.lock")):
        dataloader = get_data_loader()

    step = 1
    if session.get_checkpoint():
        checkpoint_dict = session.get_checkpoint().to_dict()
        netD.load_state_dict(checkpoint_dict["netDmodel"])
        netG.load_state_dict(checkpoint_dict["netGmodel"])
        optimizerD.load_state_dict(checkpoint_dict["optimD"])
        optimizerG.load_state_dict(checkpoint_dict["optimG"])
        # Note: Make sure to increment the loaded step by 1 to get the
        # current step.
        last_step = checkpoint_dict["step"]
        step = last_step + 1

        # NOTE: It's important to set the optimizer learning rates
        # again, since we want to explore the parameters passed in by PBT.
        # Without this, we would continue using the exact same
        # configuration as the trial whose checkpoint we are exploiting.
        if "netD_lr" in config:
            for param_group in optimizerD.param_groups:
                param_group["lr"] = config["netD_lr"]
        if "netG_lr" in config:
            for param_group in optimizerG.param_groups:
                param_group["lr"] = config["netG_lr"]

    while True:
        lossG, lossD, is_score = train(
            netD,
            netG,
            optimizerG,
            optimizerD,
            criterion,
            dataloader,
            step,
            device,
            config["mnist_model_ref"],
        )
        checkpoint = None
        if step % config["checkpoint_interval"] == 0:
            checkpoint = Checkpoint.from_dict(
                {
                    "netDmodel": netD.state_dict(),
                    "netGmodel": netG.state_dict(),
                    "optimD": optimizerD.state_dict(),
                    "optimG": optimizerG.state_dict(),
                    "step": step,
                }
            )
        session.report(
            {"lossg": lossG, "lossd": lossD, "is_score": is_score},
            checkpoint=checkpoint,
        )
        step += 1