from enum import Enum
import tensorflow as tf
tf.nn.cross
class CostFunctions(Enum):
    cross_entropy_softmax = 1
    cross_entropy_sigmoid = 2
    cross_entropy_sparse = 3
    cross_entropy_weighted = 4

