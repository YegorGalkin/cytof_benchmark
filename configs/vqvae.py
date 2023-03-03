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

    config.embed_dim1 = 2
    config.embed_entries1 = 16

    config.embed_dim2 = 4
    config.embed_entries2 = 16

    config.embed_dim3 = 5
    config.embed_entries3 = 32

    config.n_layers = 3
    config.kld_scale = 5e-4
    config.temperature = 1
    config.straight_through = False

    return config
