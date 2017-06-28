import tensorflow as tf
import time

import OptimizerFunctions as OF
import Util
import NetworkCreator as NC
import Input
import Facial_Emotion_Recognition as FER
import PerformanceMeasures as PM
import FilePaths
import TensorGraph as tg

optimizer = object


# initialize global optimizer with the required optimizer function to minimise cost.
# optimizer_string : string containing optimizer function to be used. String according to Util.get_optimizer_function(string)
def initialize_optimizer(optimizer_string):
    global optimizer
    optimizer_type = Util.get_optimizer_function(optimizer_string)
    if optimizer_type == OF.OptimizerFunctions.AdamDescent:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(NC.cost)


# function to optimize the CNN.
# number_of_iterations: number of trainin/optimization steps to be taken
# batch_size: number of training data to be taken in a set. Should be less than the training set size
def optimize(number_of_iterations, batch_size):
    print("running optimize operation")
    data = Input.read_data(FilePaths.inp_dir_path)
    start_time = time.time()
    print("Start time:", start_time)
    for i in range(number_of_iterations):
        x_batch, y_batch = data.Train.next_batch(batch_size)
        # the dict to pass values to placeholders defined in TensorGraph. Names should be same. Here {x,y}
        train_dict = {tg.x: x_batch, tg.y: y_batch}
        FER.session.run(optimizer, feed_dict=train_dict)
        # print the accuracy every 100 iterations
        if i % 2 == 0:
            out_msg = "Optimization Iteration : {0} Accuracy : {1} "
            acc = FER.session.run(PM.accuracy, feed_dict=train_dict)
            print(out_msg.format(i, acc))

    end_time = time.time()
    # time to train the data
    opt_time = end_time - start_time
    print("time to train : ", opt_time)





