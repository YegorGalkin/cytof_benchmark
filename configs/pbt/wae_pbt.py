from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    # VAE architecture parameters
    config.in_features = 41
    config.latent_dim = 2
    config.hidden_dims = [256, 256, 256, 256, 256]
    config.kld_weight = 0.0025
    config.activation = 'LeakyReLU'
    config.device = 'cuda'

    config.reg_weight = 1
    config.latent_var = 1
    config.kernel_type = 'imq'

    return config