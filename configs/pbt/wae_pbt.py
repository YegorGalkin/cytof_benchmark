from configs.pbt import base_pbt


def get_config():
    config = base_pbt.get_config()

    config.model = "WAE_MMD"
    config.latent_dim = 2
    config.bs_upper = 2 ** 3

    config.reg_weight = 1
    config.latent_var = 1
    config.kernel_type = 'imq'

    config.local_dir = 'logs/ray_tune/test_all/wae_mmd'
    config.run_name = 'wae_mmd_training'
    return config
