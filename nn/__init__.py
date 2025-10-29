import flax.linen as nn

import config.nn
from config.nn import (
    CAREConfig,
    FiLMConfig,
    MOOREConfig,
    MultiHeadConfig,
    PaCoConfig,
    SoftModulesConfig,
    VanillaNetworkConfig,
)

from .base import  VanillaNetwork
from .care import CARENetwork
from .film import FiLMNetwork
from .moore import MOORENetwork
from .multi_head import MultiHeadNetwork
from .paco import PaCoNetwork
from .soft_modules import SoftModularizationNetwork


def get_nn_arch_for_config(
    config: config.nn.NeuralNetworkConfig,
) -> type[nn.Module]:
    if type(config) is MultiHeadConfig:
        return MultiHeadNetwork
    elif type(config) is SoftModulesConfig:
        return SoftModularizationNetwork
    elif type(config) is PaCoConfig:
        return PaCoNetwork
    elif type(config) is CAREConfig:
        return CARENetwork
    elif type(config) is FiLMConfig:
        return FiLMNetwork
    elif type(config) is MOOREConfig:
        return MOORENetwork
    elif type(config) is VanillaNetworkConfig:
        return VanillaNetwork
    else:
        raise ValueError(
            f"Unknown config type: {type(config)}. (NeuralNetworkConfig by itself is not supported, use VanillaNetworkConfig)"
        )


__all__ = ["VanillaNetwork", "MultiHeadNetwork", "SoftModularizationNetwork"]
