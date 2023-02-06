import os.path

from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    # VAE architecture parameters
    config.latent_dim = 2
    config.hidden_dims = (256, 256, 256, 256, 256)
    config.kld_weight = 0.0025
    config.loss_type = 'beta'
    config.activation = 'LeakyReLU'
    config.model = "BetaVAE"
    config.device = 'cuda'
    # 41 for Organoid, 44 for CAF, 37 for Breast cancer
    config.in_features = 41
    config.dataset = 'OrganoidDataset'
    config.data_dir = '../data/organoids'
    config.data_cache = '/home/egor/vae_dataset_cache'

    config.drop_last = True

    config.local_dir = '../logs/ray_tune/test3'
    config.run_name = 'vae_training'

    config.perturbation_interval = 10
    config.perturbation_factors = (1.2, 0.8)
    config.lr_lower = 1e-9
    config.lr_upper = 1e-1

    config.bs_lower = 2 ** 10
    config.bs_upper = 2 ** 15

    config.population = 32
    config.concurrent = 32
    config.gpus = 2.0
    config.cpus = 128.0
    config.gpu_per_trial = config.gpus / config.concurrent
    config.cpus_per_trial = config.cpus / config.concurrent
    config.synch = False

    config.max_failures = 5
    config.soft_time_limit = 60 * 60 * 8
    return config
