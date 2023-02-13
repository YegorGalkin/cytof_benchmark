from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    # Architecture parameters
    config.hidden_dims = (256, 256, 256, 256, 256)
    config.activation = 'LeakyReLU'

    # 41 for Organoid, 44 for CAF, 37 for Breast cancer
    config.in_features = 41
    config.dataset = 'OrganoidDataset'
    config.data_dir = 'data/organoids'

    config.drop_last = True
    config.device = 'cuda'

    config.perturbation_interval = 100
    config.perturbation_factors = (1.5, 0.6)
    config.lr_lower = 1e-9
    config.lr_upper = 1e-1

    config.bs_lower = 2 ** 1
    config.bs_upper = 2 ** 5

    config.group_size = 2**10

    config.population = 16
    config.concurrent = 16
    config.gpus = 2.0
    config.cpus = 128.0
    config.gpu_per_trial = config.gpus / config.concurrent
    config.cpus_per_trial = config.cpus / config.concurrent
    config.synch = False

    config.max_failures = 5
    config.soft_time_limit = 60 * 60 * 8
    return config
