from .base import *
from .beta_vae import *
from .wae_mmd import *

# Aliases
VAE = BetaVAE
GaussianVAE = BetaVAE

vae_models = {
    'VanillaVAE': BetaVAE,
    'WAE_MMD': WAE_MMD,
}