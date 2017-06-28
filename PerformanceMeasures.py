import tensorflow as tf
import TensorGraph as TG
import NetworkCreator as NC

correct_prediction = object
accuracy = object

def initialize_performance_measures():
    global correct_prediction, accuracy
    correct_prediction = tf.equal(TG.y_class, NC.y_pred_class)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




