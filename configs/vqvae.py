from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    # General parameters
    config.dataset = 'Organoid'
    config.model = 'VQVAE'
    config.seed = 12345
    config.batch_size = 32*1024
    config.learning_rate = 0.001
    config.weight_decay = 0.0
    config.epochs = 200
    config.output_dir = './logs/VQVAE/'
    config.device = 'cuda'
    # VAE architecture parameters
    config.in_features = 41
    config.hidden_features = 64

    config.embed_dim = 8
    config.embed_entries = 256
    config.embed_channels = 1

    config.n_layers = 3
    config.kld_scale = 5e-4
    config.temperature = 1
    config.straight_through = False

    return config
