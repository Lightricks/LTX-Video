from enum import Enum, auto


class SkipLayerStrategy(Enum):
    Attention = auto()
    Residual = auto()
