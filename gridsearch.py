import os
from itertools import product
from multiprocessing import Pool

N_GPUS = 2

def experiment_1():
    hidden_dims = ['"(41,41,41,41,41)"', '"(64,64,64,64,64)"', '"(128,128,128,128,128)"', '"(256,256,256,256,256)"']
    learning_rates = [0.003, 0.004, 0.005, 0.006, 0.007]
    latent_dim = [2]
    with Pool(processes=8) as pool:
        commands = []
        for i, (ldim, hdim, lr) in enumerate(product(latent_dim, hidden_dims, learning_rates)):
            commands.append(f'python3 /data/PycharmProjects/cytof_benchmark/experiment.py '
                            f'--config=configs/beta_vae.py '
                            f'--config.output_dir=logs/BetaVAE/exp_1/ '
                            f'--config.hidden_dims={hdim} '
                            f'--config.learning_rate={lr} '
                            f'--config.epochs=3000 '
                            f'--config.latent_dim={ldim} '
                            f'--config.batch_size=16384 '
                            f'--config.max_grad_norm=1 '
                            f'--config.device=cuda:{i % N_GPUS} ')
        pool.map(os.system, commands)


def experiment_2():
    learning_rates = [0.003, 0.005, 0.007]
    activation = ['LeakyReLU', 'GELU', 'SiLU']
    max_grad_norm = [float('Inf'), 1]
    with Pool(processes=8) as pool:
        commands = []
        for i, (act, mgn, lr) in enumerate(product(activation, max_grad_norm, learning_rates)):
            commands.append(f'python3 /data/PycharmProjects/cytof_benchmark/experiment.py '
                            f'--config=configs/beta_vae.py '
                            f'--config.output_dir=logs/BetaVAE/exp_2/ '
                            f'--config.hidden_dims="(128,128,128,128,128)" '
                            f'--config.learning_rate={lr} '
                            f'--config.epochs=3000 '
                            f'--config.activation={act} '
                            f'--config.batch_size=16384 '
                            f'--config.max_grad_norm={mgn} '
                            f'--config.device=cuda:{i % N_GPUS} ')
        pool.map(os.system, commands)


def experiment_3():
    layers = [2, 3, 4, 5, 6, 7, 8, 9]
    hidden_dims = ['"' + str(tuple([128] * i)) + '"' for i in layers]
    with Pool(processes=8) as pool:
        commands = []
        for i, hdim in enumerate(hidden_dims):
            commands.append(f'python3 /data/PycharmProjects/cytof_benchmark/experiment.py '
                            f'--config=configs/beta_vae.py '
                            f'--config.output_dir=logs/BetaVAE/exp_3/ '
                            f'--config.hidden_dims={hdim} '
                            f'--config.learning_rate=0.005 '
                            f'--config.epochs=3000 '
                            f'--config.batch_size=16384 '
                            f'--config.device=cuda:{i % N_GPUS} ')
        pool.map(os.system, commands)


def experiment_4():
    epochs = [1000, 3000, 5000, 10000, 20000]
    with Pool(processes=8) as pool:
        commands = []
        for i, ep in enumerate(epochs):
            commands.append(f'python3 /data/PycharmProjects/cytof_benchmark/experiment.py '
                            f'--config=configs/beta_vae.py '
                            f'--config.output_dir=logs/BetaVAE/exp_4/ '
                            f'--config.hidden_dims="(128,128,128,128,128)" '
                            f'--config.learning_rate=0.005 '
                            f'--config.epochs={ep} '
                            f'--config.batch_size=16384 '
                            f'--config.max_grad_norm=1 '
                            f'--config.device=cuda:{i % N_GPUS} ')
        pool.map(os.system, commands)


def experiment_5():
    width = [2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11]
    hidden_dims = ['"' + str(tuple([i] * 5)) + '"' for i in width]
    with Pool(processes=8) as pool:
        commands = []
        for i, hdim in enumerate(hidden_dims):
            commands.append(f'python3 /data/PycharmProjects/cytof_benchmark/experiment.py '
                            f'--config=configs/beta_vae.py '
                            f'--config.output_dir=logs/BetaVAE/exp_5/ '
                            f'--config.hidden_dims={hdim} '
                            f'--config.learning_rate=0.001 '
                            f'--config.epochs=3000 '
                            f'--config.batch_size=16384 '
                            f'--config.device=cuda:{i % N_GPUS} ')
        pool.map(os.system, commands)


def experiment_6():
    width = [2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9]
    hidden_dims = ['"' + str(tuple([i] * 5)) + '"' for i in width]
    with Pool(processes=8) as pool:
        commands = []
        for i, hdim in enumerate(hidden_dims):
            commands.append(f'python3 /data/PycharmProjects/cytof_benchmark/experiment.py '
                            f'--config=configs/beta_vae.py '
                            f'--config.output_dir=logs/BetaVAE/exp_6/ '
                            f'--config.hidden_dims={hdim} '
                            f'--config.learning_rate=0.001 '
                            f'--config.epochs=3000 '
                            f'--config.activation=GELU '
                            f'--config.batch_size=16384 '
                            f'--config.device=cuda:{i % N_GPUS} ')
        pool.map(os.system, commands)


def experiment_7():
    width = [2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9]
    hidden_dims = ['"' + str(tuple([i] * 5)) + '"' for i in width]
    with Pool(processes=8) as pool:
        commands = []
        for i, hdim in enumerate(hidden_dims):
            commands.append(f'python3 /data/PycharmProjects/cytof_benchmark/experiment.py '
                            f'--config=configs/beta_vae.py '
                            f'--config.output_dir=logs/BetaVAE/exp_7/ '
                            f'--config.hidden_dims={hdim} '
                            f'--config.learning_rate=0.001 '
                            f'--config.epochs=3000 '
                            f'--config.activation=SiLU '
                            f'--config.batch_size=16384 '
                            f'--config.device=cuda:{i % N_GPUS} ')
        pool.map(os.system, commands)


if __name__ == '__main__':
    experiment_7()
