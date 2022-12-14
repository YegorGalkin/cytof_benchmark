from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    # General parameters
    config.dataset = 'Organoid'
    config.model = 'VAE'
    config.seed = 12345
    config.batch_size = 4096
    config.output_dir = './logs/VanillaVAE/'
    # VAE parameters
    config.in_features = 41
    config.latent_dim = 2
    config.hidden_dims = (32, 32, 32)

    config.kld_weight = 0.0025
    config.C_max = 25
    config.loss_type = 'disentangled_beta'

    # Optimizer and runner parameters
    config.learning_rate = 0.05
    config.weight_decay = 0.0
    config.epochs = 10000
    config.save_loss_every_n_epochs = 10
    config.device = 'cuda'
    config.C_stop_iter = config.epochs // 5
    config.activation = 'LeakyReLU'
    config.max_grad_norm = float("inf")
    return config
