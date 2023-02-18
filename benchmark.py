import os

if __name__ == '__main__':
    configs = ['configs/pbt/beta_vae_pbt.py',
               'configs/pbt/dbeta_vae_pbt.py',
               'configs/pbt/wae_pbt.py',
               'configs/pbt/hs_vae_pbt.py']
    models = ["BetaVAE", "DBetaVAE", "WAE_MMD", "HyperSphericalVAE"]

    datasets = ['OrganoidDataset', 'CafDataset', 'ChallengeDataset']
    data_dirs = ['data/organoids', 'data/caf', 'data/breast_cancer_challenge']
    features = [41, 44, 37]
    intervals = [100, 5, 15]

    for model, config in zip(models, configs):
        for dataset, data_dir, in_features, interval in zip(datasets, data_dirs, features, intervals):
            os.system(f'python3 pbt_runner.py '
                      f'--config={config} '
                      f'--config.dataset={dataset} '
                      f'--config.data_dir={data_dir} '
                      f'--config.in_features={in_features} '
                      f'--config.perturbation_interval={interval} '
                      f'--config.local_dir={os.path.join("/home/egor/Desktop/ray_tune/pbt_bench/", model, dataset)} '
                      f'--config.soft_time_limit=28800 ')
