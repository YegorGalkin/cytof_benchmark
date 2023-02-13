from configs.pbt import base_pbt


def get_config():
    config = base_pbt.get_config()

    config.model = "BetaVAE"
    config.latent_dim = 2
    config.kld_weight = 0.0025
    config.loss_type = 'disentangled_beta'
    config.C = 25

    config.local_dir = 'logs/ray_tune/test_all/dbeta_vae'
    config.run_name = 'dbeta_vae_training'
    return config