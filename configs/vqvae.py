from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    # General parameters
    config.dataset = 'Organoid'
    config.model = 'VQVAE'
    config.seed = 12345
    config.batch_size = 32*1024
    config.output_dir = './logs/VQVAE/'
    # VAE architecture parameters
    config.in_features = 41
    config.hidden_features = 32
    config.embed_dim = 16
    config.n_layers = 3
    config.beta = 0.25
    config.nb_entries = 256
    config.decay = 0.99
    config.eps = 1e-5

    return config