from ml_collections import config_dict
import numpy as np


def get_config():
    config = config_dict.ConfigDict()

    # 41 for Organoid, 44 for CAF, 37 for Breast cancer
    config.in_features = 41
    config.dataset = 'OrganoidDataset'
    config.data_dir = 'data/organoids'

    config.drop_last = True
    config.device = 'cuda'

    config.perturbation_interval = 10
    config.lrs = tuple(np.power(10, np.linspace(-2, -7, 26)))
    config.bss = tuple(int(x) for x in np.power(2, np.linspace(0, 5, 6)))
    config.kld_scales = tuple(np.power(10, np.linspace(-3, -6, 16)))
    config.temperatures = tuple(np.power(2, np.linspace(-2, 2, 13)))

    config.group_size = 2 ** 10

    config.population = 16
    config.concurrent = 16
    config.gpus = 2.0
    config.cpus = 128.0
    config.gpu_per_trial = config.gpus / config.concurrent
    config.cpus_per_trial = config.cpus / config.concurrent
    config.synch = False

    config.hidden_features = 128

    config.embed_dim1 = 8
    config.embed_entries1 = 256

    config.embed_dim2 = 8
    config.embed_entries2 = 256

    config.embed_dim3 = 8
    config.embed_entries3 = 256

    config.n_layers = 5
    config.straight_through = False

    config.max_failures = 0
    config.soft_time_limit = 60 * 60 * 8

    config.local_dir = 'logs/ray_tune/test/vqvae'
    config.run_name = 'vq_vae_training'

    return config
