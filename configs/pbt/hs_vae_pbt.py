from configs.pbt import base_pbt


def get_config():
    config = base_pbt.get_config()

    config.model = "HyperSphericalVAE"
    config.latent_dim = 3
    config.kld_weight = 0.0025

    config.local_dir = 'logs/ray_tune/test_all/hs_vae'
    config.run_name = 'hs_vae_training'

    return config