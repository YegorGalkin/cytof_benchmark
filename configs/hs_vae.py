from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    # General parameters
    config.dataset = 'Organoid'
    config.model = 'HyperSphericalVAE'
    config.seed = 12345
    config.batch_size = 2**10
    config.output_dir = './logs/HyperSphericalVAE/'
    # VAE parameters
    config.in_features = 41
    config.latent_dim = 3
    config.hidden_dims = (64, 64, 64)

    config.kld_weight = 0.0025

    # Optimizer and runner parameters
    config.learning_rate = 0.005
    config.weight_decay = 0.0
    config.epochs = 1000
    config.save_loss_every_n_epochs = 10
    config.device = 'cuda'

    config.max_grad_norm = float("inf")
    return config
