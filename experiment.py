import datetime
import math
import os.path
from glob import glob
import torch
from pytorch_lightning import seed_everything
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from datasets import OrganoidDatasetDeprecated, CellType
from models import BetaVAE, WAE_MMD, HyperSphericalVAE
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from absl import app
from ml_collections.config_flags import config_flags
import pickle

matplotlib.use('Agg')
matplotlib.style.use('ggplot')

_CONFIG = config_flags.DEFINE_config_file('config')


def main(_):
    config = _CONFIG.value

    print(f"CUDA available:{torch.cuda.is_available()}")

    if config.dataset == 'Organoid':
        data = OrganoidDatasetDeprecated(device=config.device)
    else:
        return

    seed_everything(config.seed)

    if config.model == 'VAE':
        model = BetaVAE(config=config).to(config.device)
    elif config.model == "WAE_MMD":
        model = WAE_MMD(config=config).to(config.device)
    elif config.model == "HyperSphericalVAE":
        model = HyperSphericalVAE(config=config).to(config.device)
    else:
        return

    X_train, y_train = data.train
    X_val, y_val = data.val

    X_train_batches = torch.split(X_train, split_size_or_sections=config.batch_size)
    if X_train_batches[-1].shape[0] < config.batch_size // 2:
        print("Last batch of size {} is smaller than half batch size {} and is dropped".
              format(X_train_batches[-1].shape[0], config.batch_size))
        X_train_batches = X_train_batches[:-1]

    X_val_batches = torch.split(X_val, split_size_or_sections=config.batch_size)

    optimizer = optim.Adam(model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay,
                           )
    scheduler = lr_scheduler.OneCycleLR(optimizer,
                                        max_lr=config.learning_rate,
                                        steps_per_epoch=len(X_train_batches),
                                        epochs=config.epochs)
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
            if not math.isinf(config.max_grad_norm):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

        if epoch % config.save_loss_every_n_epochs == 0 or epoch == config.epochs:
            with torch.no_grad():
                model.eval()
                train_losses = list()
                for X_batch in X_train_batches:
                    loss_dict = dict()
                    outputs = model.forward(X_batch)
                    loss_train = model.loss_function(*outputs)
                    for key in loss_train.keys():
                        loss_dict['train_' + key] = loss_train[key].to('cpu').numpy().item() * X_batch.shape[0]
                    train_losses.append(loss_dict)

                val_losses = list()
                for X_batch in X_val_batches:
                    loss_dict = dict()
                    outputs = model.forward(X_batch)
                    loss_val = model.loss_function(*outputs)
                    for key in loss_val.keys():
                        loss_dict['val_' + key] = loss_val[key].to('cpu').numpy().item() * X_batch.shape[0]
                    val_losses.append(loss_dict)

                mean_train_loss = {'train_' + key: sum(d['train_' + key] for d in train_losses) / X_train.shape[0] for
                                   key in loss_val.keys()}
                mean_val_loss = {'val_' + key: sum(d['val_' + key] for d in val_losses) / X_val.shape[0] for key in
                                 loss_val.keys()}
                loss_list.append({'epoch': epoch} | mean_train_loss | mean_val_loss)

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

    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        print(f'memory,{torch.cuda.memory_allocated()}', file=f)
        print(f'max_memory,{torch.cuda.max_memory_allocated()}', file=f)
        print(f'time,{(datetime.datetime.now() - start_time).seconds}', file=f)
        print(f'time_str,{(datetime.datetime.now() - start_time)}', file=f)
        print('val_mse,{}'.format(pd.DataFrame(loss_list)['val_MSE'].iat[-1]), file=f)

    with open(os.path.join(save_dir, 'architecture.txt'), 'w') as f:
        print(model, file=f)

    with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
        print(config, file=f)

    with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, file=f)

    latent_vals = []
    with torch.no_grad():
        for X_batch in X_val_batches:
            latent_val_batch = model.latent(X_batch).to('cpu')
            latent_vals.append(latent_val_batch)

    latent_val = torch.cat(latent_vals)
    torch.cuda.empty_cache()

    latent_df = pd.DataFrame(latent_val.numpy(), columns=["VAE{}".format(i) for i in range(1, latent_val.shape[1] + 1)])
    latent_df.to_csv(os.path.join(save_dir, 'latent.csv.gz'), index=None)

    metadata = pd.DataFrame(y_val.cpu()).rename({0: "Cell type", 1: "Day"}, axis=1)
    metadata['Cell type'].replace({i.value: i.name for i in CellType}, inplace=True)

    dot_plot = sns.lmplot(x="VAE1", y="VAE2",
                          data=pd.concat([latent_df, metadata], axis=1).head(10000),
                          fit_reg=False,
                          hue='Cell type',
                          legend=True,
                          scatter_kws={"s": 5})
    plt.savefig(os.path.join(save_dir, 'latent.png'))
    plt.clf()

    train_loss_df = pd.DataFrame(loss_list)[['epoch', 'train_MSE']] \
        .rename({'train_MSE': "Reconstruction Loss"}, axis=1) \
        .tail(int(config.epochs / config.save_loss_every_n_epochs * 0.7))
    train_loss_df['Stage'] = 'Training'

    val_loss_df = pd.DataFrame(loss_list)[['epoch', 'val_MSE']] \
        .rename({'val_MSE': "Reconstruction Loss"}, axis=1) \
        .tail(int(config.epochs / config.save_loss_every_n_epochs * 0.7))
    val_loss_df['Stage'] = 'Validation'

    loss_plot = sns.lineplot(x="epoch", y="Reconstruction Loss", hue='Stage',
                             data=pd.concat([train_loss_df, val_loss_df], ignore_index=True, axis=0),
                             legend=True)

    plt.savefig(os.path.join(save_dir, 'loss.png'))

    # UMAP plot
    if config.latent_dim > 2:
        import umap
        import umap.plot
        from bokeh.io import save

        mapper = umap.UMAP().fit(latent_df)
        umap_points = 50000
        mapper.embedding_ = mapper.embedding_[:umap_points]

        p = umap.plot.interactive(mapper,
                                  labels=metadata['Cell type'].head(umap_points),
                                  hover_data=metadata.head(umap_points),
                                  point_size=2,
                                  interactive_text_search=True,
                                  interactive_text_search_columns=['Cell type', "Day"])
        save(p, filename=os.path.join(save_dir, 'umap_2d.html'))

    # Spherical plots
    if config.model == 'HyperSphericalVAE' and config.latent_dim == 3:
        import cartopy
        import geopandas
        from shapely.geometry import Point
        import plotly.express as px

        sphere = geopandas.GeoSeries(latent_df.apply(Point, axis=1), crs=cartopy.crs.Geocentric())
        projected = sphere.to_crs(cartopy.crs.Mollweide())
        data = pd.DataFrame({'x': projected.x,
                             'y': projected.y,
                             'Cell type': metadata['Cell type'],
                             'Day': metadata['Day']})
        # 2d projection plot
        sns.lmplot(x="x", y="y",
                   data=data.head(10000),
                   fit_reg=False,
                   hue='Cell type',  # color by cluster
                   legend=True,
                   scatter_kws={"s": 5})
        plt.axis('equal')
        plt.savefig(os.path.join(save_dir, '2d_projection.png'))

        # 3d point cloud plot
        plot_data = pd.concat([latent_df, metadata], axis=1).head(5000)
        fig = px.scatter_3d(plot_data, x='VAE1', y='VAE2', z='VAE3', color='Cell type')
        fig.update_traces(marker_size=3)
        fig.write_html(os.path.join(save_dir, '3d_plot.html'))


if __name__ == '__main__':
    app.run(main)
