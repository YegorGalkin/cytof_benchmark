import copy
import os.path
from functools import partial

import numpy as np
import ray
import torch
import torch.optim as optim
from ml_collections.config_flags import config_flags
from absl import app
from ray import tune, air
from ray.air import Checkpoint
from ray.tune import Trainable
from ray.tune.schedulers import HyperBandForBOHB
import time

from ray.tune.search.bohb import TuneBOHB
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
from torch.utils.data import TensorDataset, DataLoader

import datasets
import models


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


class TrainableVAE(Trainable):

    def setup(self, ray_cfg):
        self.last_step = 0
        self.config = copy.deepcopy(ray.get(ray_cfg['config']))
        with self.config.unlocked():
            self.config.hidden_dims = tuple([ray_cfg['width']] * ray_cfg['n_layers'])

        self.time_start = ray_cfg["time_start"]
        self.model_class = getattr(models, self.config.model)

        self.model = self.model_class(self.config).to(self.config.device)

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=ray_cfg["lr"],
                                    # betas=(1-ray_cfg['beta1'], 1-ray_cfg['beta2']),
                                    # eps=ray_cfg['eps']
                                    )



        self.X_train = ray.get(ray_cfg["train_dataset"])
        self.X_val = ray.get(ray_cfg["val_dataset"])

        self.X_train_dl = DataLoader(self.X_train, batch_size=ray_cfg["batch_size"], drop_last=self.config.drop_last,
                                     pin_memory=True)
        self.X_val_dl = DataLoader(self.X_val, batch_size=self.config.bs_upper, drop_last=False, pin_memory=True)

        self.scheduler = OneCycleLR(optimizer=self.optimizer,
                                    max_lr=ray_cfg["lr"],
                                    pct_start=ray_cfg["pct_start"],
                                    epochs=10000,
                                    steps_per_epoch=len(self.X_train_dl))

    def step(self):
        metrics = None
        for i in range(self.config.eval_interval):
            self.last_step += 1
            train(self.model, self.config, self.optimizer, self.scheduler, self.X_train_dl)

            if not metrics:
                metrics = test(self.model, self.config, self.X_val_dl)
            else:
                #EMA loss over 100 epochs
                results = test(self.model, self.config, self.X_val_dl)
                for key in metrics:
                    metrics[key] = 0.1*results[key]+0.9*metrics[key]

        checkpoint_time = time.time() - self.time_start
        return metrics | {
            "step": self.last_step,
            "checkpoint_time": checkpoint_time,
        }

    def save_checkpoint(self, checkpoint_dir):
        checkpoint = Checkpoint.from_dict(
            {
                "model": self.model.state_dict(),
                "optim": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self.last_step,
            }
        )
        checkpoint.to_directory(checkpoint_dir)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path):
        checkpoint_dict = Checkpoint.from_directory(checkpoint_path).to_dict()

        self.model.load_state_dict(checkpoint_dict["model"])
        self.optimizer.load_state_dict(checkpoint_dict["optim"])
        self.scheduler.load_state_dict(checkpoint_dict["scheduler"])
        self.last_step = checkpoint_dict["step"] + 1


def train(model, config, optimizer, scheduler, train_dataloader):
    for X_batch in train_dataloader:
        X_batch = X_batch[0].flatten(end_dim=1).to(config.device, non_blocking=True)
        optimizer.zero_grad()
        model.train()
        outputs = model.forward(X_batch)
        loss = model.loss_function(*outputs)
        loss['loss'].backward()
        optimizer.step()
        scheduler.step()


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

    num_samples = config.num_samples
    search_space = {
        "lr": tune.loguniform(config.lr_lower, config.lr_upper),
        "pct_start": tune.loguniform(config.pct_start_lower, config.pct_start_upper),
        #"beta1": tune.loguniform(config.beta1_lower, config.beta1_upper),
        #"beta2": tune.loguniform(config.beta2_lower, config.beta2_upper),
        #"eps": tune.loguniform(config.eps_lower, config.eps_upper),
        "batch_size": tune.lograndint(config.bs_lower, config.bs_upper, base=2),
        "activation": tune.choice(config.activations),
        "n_layers": tune.randint(config.layer_min, config.layer_max),
        "width": tune.lograndint(config.width_min, config.width_max, base=2)
    }

    algo = TuneBOHB()
    algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=config.concurrent)

    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=100,
        stop_last_trials=False,
    )

    tuner = tune.Tuner(
        tune.with_resources(TrainableVAE, {"cpu": config.cpus_per_trial,
                                           "gpu": config.gpu_per_trial}),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=algo,
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            failure_config=ray.air.config.FailureConfig(max_failures=config.max_failures),
            local_dir=config.local_dir,
            name=config.run_name,
            stop={"checkpoint_time": config.soft_time_limit},
            verbose=2,
        ),
        param_space=search_space | {
            # Define how initial values of the learning rates should be chosen.
            "time_start": time.time(),
            "train_dataset": ray.put(X_train_ds),
            "val_dataset": ray.put(X_val_ds),
            "config": ray.put(config),
        },
    )
    results_grid = tuner.fit()

    best_result = results_grid.get_best_result(metric="loss", mode="min")

    with open(os.path.join(config.local_dir, 'summary.txt'), 'w') as f:
        for key in best_result.metrics:
            print('{},{}'.format(key, best_result.metrics[key]), file=f)

    torch.save(best_result.checkpoint.to_dict(), os.path.join(config.local_dir, 'model.pth'))

    ray.shutdown()


if __name__ == '__main__':
    app.run(main)
