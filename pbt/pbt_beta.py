import numpy as np
import ray
import torch
import torch.optim as optim
from ray import tune, air
from ray.air import session, Checkpoint
from ray.tune.schedulers import PopulationBasedTraining
import time

from datasets import OrganoidDataset
from configs.pbt.beta_vae_pbt import get_config
from models import BetaVAE


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
                loss_dict[key] = losses[key].to('cpu').numpy().item() * X_batch.shape[0]
            mse.append(loss_dict['MSE'])
            kld.append(loss_dict['KLD'])
            loss.append(loss_dict['loss'])
    data_len = sum(len(batch) for batch in val_dataloader)

    return \
        np.nan_to_num(sum(mse) / data_len, nan=10), \
        np.nan_to_num(sum(kld) / data_len, nan=10), \
        np.nan_to_num(sum(loss) / data_len, nan=10)


def vae_train(cfg):
    config = get_config()
    model = BetaVAE(config).to(config.device)

    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.get("learning_rate"),
                           )

    dataset = OrganoidDataset(data_dir='/data/PycharmProjects/cytof_benchmark/data/organoids', device=config.device)
    X_train, y_train = dataset.train
    X_val, y_val = dataset.val
    X_train_batches = torch.split(X_train, split_size_or_sections=cfg.get("batch_size"))
    X_val_batches = torch.split(X_val, split_size_or_sections=cfg.get("batch_size"))

    # Remove last batch if it has less than half batch size samples to reduce variance
    if X_train_batches[-1].shape[0] < cfg.get("batch_size") // 2:
        X_train_batches = X_train_batches[:-1]

    # Initialize step count and checkpoint timer
    checkpoint_time = 0.0
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
                param_group["lr"] = cfg["learning_rate"]
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
            checkpoint_time = time.time()-cfg.get("time_start")
        session.report(
            {
                "MSE": MSE,
                "KLD": KLD,
                "loss": loss,
                'lr': cfg.get("learning_rate"),
                "step": step,
                "checkpoint_time": checkpoint_time,
            },
            checkpoint=checkpoint,
        )
        step += 1


if __name__ == '__main__':
    ray.init()

    def explore(config):
        config["batch_size"] = min(config["batch_size"], 32*1024)
        return config

    perturbation_interval = 50
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        perturbation_factors=(1.5, 0.6),
        hyperparam_mutations={
            # Distribution for resampling
            "learning_rate": tune.loguniform(1e-9, 1e-1),
            "batch_size": tune.qlograndint(1024, 32 * 1024, 32, base=2),
        },
        custom_explore_fn=explore,
    )

    tuner = tune.Tuner(
        tune.with_resources(vae_train, {"cpu": 4, "gpu": 1.0/16}),
        run_config=air.RunConfig(
            failure_config=ray.air.config.FailureConfig(max_failures=5),
            local_dir='/data/PycharmProjects/cytof_benchmark/logs/ray_tune/Beta_VAE',
            name="vae_training",
            stop={"checkpoint_time": 60*60*8},
            verbose=2,
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=32,
            scheduler=scheduler,
        ),
        param_space={
            # Define how initial values of the learning rates should be chosen.
            "learning_rate": tune.loguniform(1e-9, 1e-1),
            "batch_size": tune.qlograndint(1024, 32 * 1024, 32, base=2),
            "checkpoint_interval": perturbation_interval,
            "time_start": time.time(),
        },
    )
    results_grid = tuner.fit()

    best_result = results_grid.get_best_result(metric="loss", mode="min")

    torch.save(best_result.checkpoint.to_dict(), '/data/PycharmProjects/cytof_benchmark/logs/ray_tune/Beta_VAE/Beta_VAE_big.pth')
    with open('/data/PycharmProjects/cytof_benchmark/logs/ray_tune/Beta_VAE/Beta_VAE_summary.txt', 'w') as f:
        print('val_mse,{}'.format(best_result.metrics['MSE']), file=f)
        print('val_kld,{}'.format(best_result.metrics['KLD']), file=f)
        print('val_loss,{}'.format(best_result.metrics['loss']), file=f)
