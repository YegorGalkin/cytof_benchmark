import datetime
import os.path
from glob import glob
import torch
from pytorch_lightning import seed_everything
from torch import optim, autocast
from torch.optim import lr_scheduler
from tqdm import tqdm
from datasets import OrganoidDataset
from models import VQVAE
import pandas as pd
from absl import app
from ml_collections.config_flags import config_flags
import pickle

_CONFIG = config_flags.DEFINE_config_file('config')


def main(_):
    config = _CONFIG.value
    print(f"CUDA available:{torch.cuda.is_available()}")
    seed_everything(config.seed)
    data = OrganoidDataset()
    model = VQVAE(config=config).to(config.device)

    X_train, y_train = data.train
    X_val, y_val = data.val

    X_train_batches = torch.split(torch.Tensor(X_train).to(config.device), split_size_or_sections=32 * 1024)
    if X_train_batches[-1].shape[0] < 16 * 1024:
        X_train_batches = X_train_batches[:-1]

    X_val_batches = torch.split(torch.Tensor(X_val).to(config.device), split_size_or_sections=32 * 1024)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay,
                            )
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    loss_list = list()
    torch.cuda.empty_cache()
    start_time = datetime.datetime.now()
    for epoch in tqdm(range(1, config.epochs + 1)):
        for X_batch in X_train_batches:
            optimizer.zero_grad()
            model.train()
            outputs = model.forward(X_batch)
            loss = model.loss_function(*outputs)

            loss['loss'].backward()
            optimizer.step()
        scheduler.step()

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                model.eval()
                val_losses = list()
                for X_batch in X_val_batches:
                    loss_dict = dict()
                    outputs = model.forward(X_batch)
                    loss_val = model.loss_function(*outputs)
                    for key in loss_val.keys():
                        loss_dict['val_' + key] = loss_val[key].to('cpu').numpy().item() * X_batch.shape[0]
                    val_losses.append(loss_dict)

                mean_val_loss = {'val_' + key: sum(d['val_' + key] for d in val_losses) / X_val.shape[0] for key in
                                 loss_val.keys()}
                loss_list.append({'epoch': epoch} | mean_val_loss)

    print(f'Memory allocated:{torch.cuda.memory_allocated()}')
    print(f'Max memory allocated:{torch.cuda.max_memory_allocated()}')
    print(f'Finished Training in {(datetime.datetime.now() - start_time)}')
    torch.cuda.empty_cache()

    run_dirs = glob(os.path.join(config.output_dir, 'run_*'))
    max_run = max([int(os.path.basename(run_dir).split('_')[1]) for run_dir in run_dirs]) if run_dirs else 0
    save_dir = os.path.join(config.output_dir, 'run_{}'.format(max_run + 1))
    os.makedirs(save_dir, exist_ok=True)

    pd.DataFrame(loss_list).to_csv(os.path.join(save_dir, 'loss_curve.csv'), index=None)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))

    def get_parameter_count(net: torch.nn.Module) -> int:
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        print(f'memory,{torch.cuda.memory_allocated()}', file=f)
        print(f'max_memory,{torch.cuda.max_memory_allocated()}', file=f)
        print(f'time,{(datetime.datetime.now() - start_time).seconds}', file=f)
        print(f'time_str,{(datetime.datetime.now() - start_time)}', file=f)
        print('val_mse,{}'.format(pd.DataFrame(loss_list)['val_MSE'].iat[-1]), file=f)
        print(f"params,{get_parameter_count(model)}", file=f)

    with open(os.path.join(save_dir, 'architecture.txt'), 'w') as f:
        print(model, file=f)

    with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
        print(config, file=f)

    with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, file=f)


if __name__ == '__main__':
    app.run(main)
