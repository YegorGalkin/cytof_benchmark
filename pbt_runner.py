import os.path
from functools import partial

import numpy as np
import ray
import torch
import torch.optim as optim
from ml_collections.config_flags import config_flags
from absl import app
from ray import tune, air
from ray.air import session, Checkpoint
from ray.tune.schedulers import PopulationBasedTraining
import time

from torch.utils.data import TensorDataset, DataLoader

import datasets
import models


def train(model, config, optimizer, train_dataloader):
    for X_batch in train_dataloader:
        X_batch = X_batch[0].flatten(end_dim=1).to(config.device, non_blocking=True)
        optimizer.zero_grad()
        model.train()
        outputs = model.forward(X_batch)
        loss = model.loss_function(*outputs)
        loss['loss'].backward()
        optimizer.step()


def test(model, config, val_dataloader, na_replace=10):
    with torch.no_grad():
        model.eval()
        metrics = dict()
        for X_batch in val_dataloader:
            X_batch = X_batch[0].flatten(end_dim=1).to(config.device, non_blocking=True)
            outputs = model.forward(X_batch)
            losses = model.loss_function(*outputs)
            for key in losses.keys():
                metrics[key] = metrics.get(key, []) + [losses[key].to('cpu').numpy().item()]
            metrics['batch_size'] = metrics.get('batch_size', []) + [X_batch.shape[0]]

    return {key: np.nan_to_num(
        np.multiply(metrics[key], metrics['batch_size']).sum() / np.sum(metrics['batch_size']),
        nan=na_replace)
        for key in metrics.keys() if key != 'batch_size'}


def vae_train(ray_cfg, config):
    model_class = getattr(models, config.model)

    model = model_class(config).to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=ray_cfg["learning_rate"])

    X_train = ray.get(ray_cfg["train_dataset"])
    X_val = ray.get(ray_cfg["val_dataset"])

    X_train_dl = DataLoader(X_train, batch_size=ray_cfg["batch_size"], drop_last=config.drop_last, pin_memory=True)
    X_val_dl = DataLoader(X_val, batch_size=config.bs_upper, drop_last=False, pin_memory=True)

    # Initialize step count
    step = 1
    metrics = None
    if session.get_checkpoint():
        checkpoint_dict = session.get_checkpoint().to_dict()

        model.load_state_dict(checkpoint_dict["model"])
        optimizer.load_state_dict(checkpoint_dict["optim"])
        # Note: Make sure to increment the loaded step by 1 to get the
        # current step.
        last_step = checkpoint_dict["step"]
        step = last_step + 1
        metrics = None
        # NOTE: It's important to set the optimizer learning rates
        # again, since we want to explore the parameters passed in by PBT.
        # Without this, we would continue using the exact same
        # configuration as the trial whose checkpoint we are exploiting.
        if "learning_rate" in ray_cfg:
            for param_group in optimizer.param_groups:
                param_group["lr"] = ray_cfg["learning_rate"]

    while True:
        train(model, config, optimizer, X_train_dl)

        if not metrics:
            metrics = test(model, config, X_val_dl)
        else:
            # EMA validiation loss
            results = test(model, config, X_val_dl)
            # TODO: just pick the best dictionary based on loss
            for key in metrics:
                metrics[key] = min(results[key], metrics[key])

        if step % ray_cfg["checkpoint_interval"] == 0:
            checkpoint = Checkpoint.from_dict(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "step": step,
                }
            )
            checkpoint_time = time.time() - ray_cfg["time_start"]
            session.report(
                metrics |
                {
                    'lr': ray_cfg["learning_rate"],
                    "step": step,
                    "checkpoint_time": checkpoint_time,
                },
                checkpoint=checkpoint,
            )
        step += 1


_CONFIG = config_flags.DEFINE_config_file('config')


def main(_):
    config = _CONFIG.value
    ray.init()

    dataset_class = getattr(datasets, config.dataset)
    dataset = dataset_class(data_dir=config.data_dir)

    X_train, y_train = dataset.train
    X_val, y_val = dataset.val

    X_train = X_train[:X_train.shape[0] // config.group_size * config.group_size].reshape(
        (-1, config.group_size, X_train.shape[1])).copy()
    X_val = X_val[:X_val.shape[0] // config.group_size * config.group_size].reshape(
        (-1, config.group_size, X_val.shape[1])).copy()

    X_train_ds = TensorDataset(torch.Tensor(X_train))
    X_val_ds = TensorDataset(torch.Tensor(X_val))

    def explore(cfg):
        cfg["batch_size"] = int(np.clip(cfg["batch_size"], a_min=config.bs_lower, a_max=config.bs_upper))
        return cfg

    scheduler = PopulationBasedTraining(
        time_attr="step",
        perturbation_interval=config.perturbation_interval,
        perturbation_factors=config.perturbation_factors,
        hyperparam_mutations={
            # Distribution for resampling
            "learning_rate": tune.loguniform(config.lr_lower, config.lr_upper),
            "batch_size": tune.lograndint(config.bs_lower, config.bs_upper, base=2)
        },
        synch=config.synch,
        custom_explore_fn=explore,
    )

    tuner = tune.Tuner(
        tune.with_resources(partial(vae_train, config=config), {"cpu": config.cpus_per_trial,
                                                                "gpu": config.gpu_per_trial}),
        run_config=air.RunConfig(
            failure_config=ray.air.config.FailureConfig(max_failures=config.max_failures),
            local_dir=config.local_dir,
            name=config.run_name,
            stop={"checkpoint_time": config.soft_time_limit},
            verbose=2,
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=config.population,
            scheduler=scheduler,
        ),
        param_space={
            # Define how initial values of the learning rates should be chosen.
            "learning_rate": tune.loguniform(config.lr_lower, config.lr_upper),
            "batch_size": tune.lograndint(config.bs_lower, config.bs_upper, base=2),
            "checkpoint_interval": config.perturbation_interval,
            "time_start": time.time(),
            "train_dataset": ray.put(X_train_ds),
            "val_dataset": ray.put(X_val_ds),
        },
    )
    results_grid = tuner.fit()

    best_result = results_grid.get_best_result(metric="loss", mode="min")

    torch.save(best_result.checkpoint.to_dict(), os.path.join(config.local_dir, 'model.pth'))

    with open(os.path.join(config.local_dir, 'summary.txt'), 'w') as f:
        for key in best_result.metrics:
            print('{},{}'.format(key, best_result.metrics[key]), file=f)

    ray.shutdown()


if __name__ == '__main__':
    app.run(main)
