import json
import os
import TensorGraph as tg
import ActivationFunctions as AF
import tensorflow as tf
import Util

# global variables to be used in tensorflow session
y_pred_class = object
prediction_layer = object
cost = object


# function to create network using the config in the json.
# json_path : the path to the config file
def create_network(json_path):
    global y_pred_class, prediction_layer
    conv_layers_config = []
    conv_layers = []
    weights_conv = []
    fc_layers_config = []
    fc_layers = []
    weights_fc = []
    json_file = open(json_path)
    layers = json.load(json_file)
    for layer in layers:
        type = layer["layer"]
        if type == "convolution":
            conv_layers_config.append(layer)
        elif type == "fully connected":
            fc_layers_config.append(layer)

    # sorting layers according to levels so that they can be created in order.
    conv_layers_config = sorted(conv_layers_config, key=lambda k: k["level"])
    fc_layers_config = sorted(fc_layers_config, key=lambda k: k["level"])

    # iterator for creating convolution layers. Default activation function is ReLU.
    for idx, conv_layer_config in enumerate(conv_layers_config):
        if "activation_function" in conv_layer_config:
            activation_function = Util.get_activation_function(conv_layer_config["activation_function"])
        else:
            activation_function = AF.ActivationFunctions.relu
        if len(conv_layers) == 0:
            conv_layer, weights = tg.new_convolution_layer(tg.x_img, conv_layer_config, activation_function)
        else:
            conv_layer_config["input_channels"] = conv_layers_config[idx - 1]["number_of_filters"]
            conv_layer, weights = tg.new_convolution_layer(conv_layers[-1], conv_layer_config, activation_function)
        conv_layers.append(conv_layer)
        weights_conv.append(weights)
        print("Convolution layer: ", idx + 1, " created")
    # 4-D tensor is to be flattened for input into fully connected layer which takes 2-D vector as input
    flattened_layer, num_features = tg.flatten_convolution_layer(conv_layers[-1])
    print("flattened layer created")

    for idx, fc_layer_config in enumerate(fc_layers_config):
        if "activation_function" in fc_layer_config:
            activation_function = Util.get_activation_function(fc_layer_config["activation_function"])
        else:
            activation_function = AF.ActivationFunctions.none
        if idx == 0:
            fc_layer_config["number_of_inputs"] = num_features
            fc_layer, weights = tg.new_fc_layer(flattened_layer, fc_layer_config, activation_function)
        else:
            fc_layer_config["number_of_inputs"] = fc_layers_config[idx - 1]["number_of_outputs"]
            fc_layer, weights = tg.new_fc_layer(fc_layers[idx - 1], fc_layer_config, activation_function)
        fc_layers.append(fc_layer)
        weights_fc.append(weights)
        print("Fully connected layer: ", idx + 1, "created")

    prediction_layer = fc_layers[-1]
    y_pred = tf.nn.softmax(fc_layers[-1])
    y_pred_class = tf.argmax(y_pred, dimension=1)


def define_cost():
    global cost
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction_layer, labels=tg.y)
    cost = tf.reduce_mean(cross_entropy)









