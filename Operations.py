import Optimizer
import Constants
import PerformanceMeasures
import Facial_Emotion_Recognition as FER
import tensorflow as tf


def train():
    print("Running train operation")
    Optimizer.initialize_optimizer("adam_descent")
    FER.session.run(tf.global_variables_initializer())
    Optimizer.optimize(number_of_iterations=Constants.train_iterations, batch_size=Constants.train_batch_size)
