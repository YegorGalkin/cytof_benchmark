from .base import *
from .vanilla_vae import *
from .wae_mmd import *

# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE

vae_models = {
    'VanillaVAE': VanillaVAE,
    'WAE_MMD': WAE_MMD,
}