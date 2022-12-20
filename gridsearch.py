import os
from itertools import product


def experiment_1():

    hidden_dims = ['"(41,41,41,41,41)"', '"(64,64,64,64,64)"', '"(128,128,128,128,128)"']
    learning_rates = [0.003, 0.005, 0.007]
    latent_dim = [2, 5, 10]
    for (ldim, hdim, lr) in product(latent_dim, hidden_dims, learning_rates):
        os.system(f'python3 /home/egor/PycharmProjects/deep_dr/experiment.py '
                  f'--config=configs/beta_vae.py '
                  f'--config.output_dir=logs/BetaVAE/exp_1/ '
                  f'--config.hidden_dims={hdim} '
                  f'--config.learning_rate={lr} '
                  f'--config.epochs=3000 '
                  f'--config.latent_dim={ldim} '
                  f'--config.batch_size=16384 '
                  f'--config.max_grad_norm=1 ')


def experiment_2():
    learning_rates = [0.003, 0.005, 0.007]
    activation = ['LeakyReLU', 'GELU', 'SiLU']
    max_grad_norm = [float('Inf'), 1]
    for (act, mgn, lr) in product(activation, max_grad_norm, learning_rates):
        os.system(f'python3 /home/egor/PycharmProjects/deep_dr/experiment.py '
                  f'--config=configs/beta_vae.py '
                  f'--config.output_dir=logs/BetaVAE/exp_2/ '
                  f'--config.hidden_dims="(128,128,128,128,128)" '
                  f'--config.learning_rate={lr} '
                  f'--config.epochs=3000 '
                  f'--config.activation={act} '
                  f'--config.batch_size=16384 '
                  f'--config.max_grad_norm={mgn} ')


if __name__ == '__main__':
    experiment_1()
