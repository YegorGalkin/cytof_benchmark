from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    # Architecture parameters
    config.activation = 'LeakyReLU'

    # 41 for Organoid, 44 for CAF, 37 for Breast cancer
    config.in_features = 41
    config.dataset = 'OrganoidDataset'
    config.data_dir = 'data/organoids'

    config.drop_last = True
    config.device = 'cuda'

    config.model = "BetaVAE"
    config.latent_dim = 2
    config.kld_weight = 0.0025
    config.loss_type = 'beta'

    config.local_dir = '/home/egor/Desktop/ray_tune/bohb_run1'
    config.run_name = 'beta_vae_training'

    config.lr_lower, config.lr_upper = 1e-3, 1e-1

    config.beta1_lower, config.beta1_upper = 1e-2, 0.5
    config.beta2_lower, config.beta2_upper = 1e-4, 1e-2
    config.eps_lower, config.eps_upper = 1e-10, 1e-6
    config.pct_start_lower = 0.0001
    config.pct_start_upper = 0.05

    config.bs_lower = 2 ** 1
    config.bs_upper = 2 ** 5

    config.activations = ('LeakyReLU', 'GELU', 'SiLU')

    config.layer_min = 2
    config.layer_max = 10

    config.width_min = 32
    config.width_max = 256

    config.group_size = 2**10

    config.concurrent = 8
    config.gpus = 2.0
    config.cpus = 128.0
    config.gpu_per_trial = config.gpus / config.concurrent
    config.cpus_per_trial = config.cpus / config.concurrent
    config.synch = False

    config.max_failures = 5
    config.eval_interval = 100
    config.soft_time_limit = 60 * 60 * 8
    config.num_samples = 100
    return config
