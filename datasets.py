from enum import Enum

import torch
from os import path
import pandas as pd
import numpy as np


class CellType(Enum):
    Enterocyte = 0
    Enteroendocrine = 1
    Goblet = 2
    Paneth = 3
    Stem = 4
    Tuft = 5


class OrganoidDataset:
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

        self.train = X[train_idx].clone().to(device), y[train_idx].clone().to(device)
        self.val = X[val_idx].clone().to(device), y[val_idx].clone().to(device)
        self.test = X[test_idx].clone().to(device), y[test_idx].clone().to(device)

        self.variables = data_df.columns
