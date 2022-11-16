from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    # General parameters
    config.dataset = 'Organoid'
    config.model = 'VAE'
    config.seed = 12345
    config.batch_size = 10**6
    config.output_dir = './logs/VanillaVAE/'
    # VAE parameters
    config.in_features = 41
    config.latent_dim = 2
    config.hidden_dims = (32, 32, 32)

    config.kld_weight = 0.0025



    # Optimizer and runner parameters
    config.learning_rate = 0.005
    config.weight_decay = 0.0
    config.epochs = 10000
    config.save_loss_every_n_epochs = 25
    config.device = 'cuda'

    return config
