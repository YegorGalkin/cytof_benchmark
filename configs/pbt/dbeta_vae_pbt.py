from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    # VAE architecture parameters
    config.in_features = 41
    config.latent_dim = 2
    config.hidden_dims = [256, 256, 256, 256, 256]
    config.kld_weight = 0.0025
    config.loss_type = 'disentangled_beta'
    config.C = 1.0

    config.activation = 'LeakyReLU'
    config.device = 'cuda'

    return config