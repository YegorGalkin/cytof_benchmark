import os.path
from enum import Enum

import torch
from os import path
import pandas as pd
import numpy as np
import glob


class CellType(Enum):
    Enterocyte = 0
    Enteroendocrine = 1
    Goblet = 2
    Paneth = 3
    Stem = 4
    Tuft = 5


class OrganoidDatasetDeprecated:
    def __init__(self, data_dir: str = "./data/organoids/", device: str = 'cuda', seed: int = 12345):
        data_df = pd.read_csv(path.join(data_dir, 'full', 'data.csv.gz'), index_col='id')

        metadata_df = pd.read_csv(path.join(data_dir, 'full', 'metadata.csv.gz'), index_col='id')
        metadata_df['cell_type'].replace({i.name: i.value for i in CellType}, inplace=True)

        # Apply hyperbolic arcsinh transform according to paper
        X = torch.from_numpy(np.arcsinh(data_df.to_numpy() / 5).astype(np.float32))
        y = torch.from_numpy(metadata_df.to_numpy().astype(np.float32))

        rng = np.random.default_rng(seed=seed)
        indices = rng.permutation(len(X))

        train_n, val_n = int(len(X) * 0.7), int(len(X) * 0.2)
        train_idx, val_idx, test_idx = indices[:train_n], indices[train_n:train_n + val_n], indices[train_n + val_n:]

        self.train = X[train_idx].clone().to(device, non_blocking=True), y[train_idx].clone().to(device,
                                                                                                 non_blocking=True)
        self.val = X[val_idx].clone().to(device, non_blocking=True), y[val_idx].clone().to(device, non_blocking=True)
        self.test = X[test_idx].clone().to(device, non_blocking=True), y[test_idx].clone().to(device, non_blocking=True)

        self.variables = data_df.columns


def prepare_splits(X: np.array, y: pd.DataFrame, seed: int, split_frac=np.array([0.7, 0.2, 0.1])):
    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(len(X))

    train_n, val_n = int(len(X) * split_frac[0]), int(len(X) * split_frac[1])
    train_idx, val_idx, test_idx = indices[:train_n], indices[train_n:train_n + val_n], indices[train_n + val_n:]

    return (
        (X[train_idx].copy(), y.iloc[train_idx].copy().reset_index()),
        (X[val_idx].copy(), y.iloc[val_idx].copy().reset_index()),
        (X[test_idx].copy(), y.iloc[test_idx].copy().reset_index()),
    )


class OrganoidDataset:
    def __init__(self, data_dir: str = "./data/organoids/", seed: int = 12345):
        data_df = pd.read_csv(path.join(data_dir, 'full', 'data.csv.gz'), index_col='id')

        metadata_df = pd.read_csv(path.join(data_dir, 'full', 'metadata.csv.gz'), index_col='id')
        metadata_df.index.rename('index', inplace=True)
        # Apply hyperbolic arcsinh transform
        X = np.arcsinh(data_df / 5).astype(np.float32).to_numpy()
        y = metadata_df

        self.train, self.val, self.test = prepare_splits(X, y, seed)
        self.variables = data_df.columns


class CafDataset:
    def __init__(self, data_dir: str = "./data/caf/", seed: int = 12345):
        data_df = pd.read_pickle(path.join(data_dir, 'Metadata_PDO_CAF_screening', 'Metadata_final_paper'))

        X_COL_LAST = 44

        X = data_df.iloc[:, :X_COL_LAST].to_numpy()
        y = data_df.iloc[:, X_COL_LAST:]

        # Apply hyperbolic arcsinh transform
        X = np.arcsinh(X / 5).astype(np.float32)

        self.train, self.val, self.test = prepare_splits(X, y, seed)
        self.variables = data_df.columns[:X_COL_LAST]


class ChallengeDataset:
    def __init__(self, data_dir: str = "./data/breast_cancer_challenge/", seed: int = 12345):
        # Dataset is already batch normalized and arcsinh preprocessed
        filenames = glob.glob(os.path.join(data_dir, "*.csv"))
        dfs = [pd.read_csv(filename) for filename in filenames]
        data_df = pd.concat(dfs)

        # Removes 6M records out of 17M. There are two biomarkers with NAs present.
        data_df.dropna(inplace=True)

        X_COL_FIRST = 5

        X = data_df.iloc[:, X_COL_FIRST:].to_numpy().astype('float32', casting='same_kind')
        y = data_df.iloc[:, :X_COL_FIRST]

        self.train, self.val, self.test = prepare_splits(X, y, seed)

        self.variables = data_df.columns[X_COL_FIRST:]
