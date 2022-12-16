from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    # General parameters
    config.dataset = 'Organoid'
    config.model = 'WAE_MMD'
    config.seed = 12345
    config.batch_size = 4096
    config.output_dir = './logs/WAE_MMD/'
    # WAE MMD parameters
    config.in_features = 41
    config.latent_dim = 2
    config.hidden_dims = (32, 32, 32)

    config.reg_weight = 1
    config.latent_var = 1
    config.kernel_type = 'imq'

    # Optimizer and runner parameters
    config.learning_rate = 0.05
    config.weight_decay = 0.0
    config.epochs = 100
    config.save_loss_every_n_epochs = 2
    config.device = 'cuda'

    config.activation = 'LeakyReLU'
    config.max_grad_norm = float("inf")

    return config
