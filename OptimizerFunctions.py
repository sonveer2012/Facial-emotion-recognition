from enum import Enum
import tensorflow as tf

class OptimizerFunctions(Enum):
    GradientDescent = 1
    AdamDescent = 2
    Momentum = 3