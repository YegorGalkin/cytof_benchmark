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
    config.kld_scale = 5e-4
    config.nb_entries = 256
    config.temperature = 1
    config.straight_through = False

    return config