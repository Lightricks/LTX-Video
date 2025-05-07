from enum import Enum, auto


class SkipLayerStrategy(Enum):
    AttentionSkip = auto()
    AttentionValues = auto()
    Residual = auto()
    TransformerBlock = auto()
