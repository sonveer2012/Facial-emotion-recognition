import tensorflow as tf
import Constants as cnst
import ActivationFunctions as AF


# returns a weight matrix
# shape : shape of weight matrix
# std_dev : deviation to initialise weight matrix
def new_weights(shape, std_dev):
    return tf.Variable(tf.truncated_normal(shape, stddev=std_dev))


# returns bias vector.
# length: length of bias vector
# initial_value: value to initialise bias vector with
def new_bias(length, initial_value):
    return tf.Variable(tf.constant(initial_value, shape=[length]))


# activate the convolution layer
# layer : the layer to be activated
# activation_function : type : enum ActivationFunctions.value
def activate(layer, activation_function):
    if activation_function == AF.ActivationFunctions.none:
        return layer
    elif activation_function == AF.ActivationFunctions.relu:
        return tf.nn.relu(layer)
    elif activation_function == AF.ActivationFunctions.sigmoid:
        return tf.nn.sigmoid(layer)
    elif activation_function == AF.ActivationFunctions.softplus:
        return tf.nn.softplus(layer)
    elif activation_function == AF.ActivationFunctions.tanh:
        return tf.nn.tanh(layer)


# creates a new convolution layer.
# input : the input to the convolution layer.
# convolution_properties : dictionary of convolution properties to be defined in config.py
# activation_function : activation function for the layer. To be defined as enum
def new_convolution_layer(input, convolution_properties, activation_function):
    # shape as defined by the tensorflow library for 2d convolution network. Assuming the filter to be square
    shape = [convolution_properties["filter_size"], convolution_properties["filter_size"],
             convolution_properties["input_channels"], convolution_properties["number_of_filters"]]

    # contains the weights of all filters for the convolution layer
    weights = new_weights(shape, std_dev=convolution_properties["std_dev"])

    # bias vector. one bias value for each filter.
    biases = new_bias(convolution_properties["number_of_filters"], initial_value=0.5)

    # strides define the shift of filter in dimensions.
    if "strides" in convolution_properties:
        strides = convolution_properties["strides"]
    else:
        strides = [1, 1, 1, 1]

    # padding defines the shape of the outpur w.r.t the input
    if "padding" in convolution_properties:
        padding = convolution_properties["padding"]
    else:
        padding = "SAME"

    # creates tensorflow defined convolution layer.
    conv_layer = tf.nn.conv2d(input, weights, strides, padding)

    # adding bias to the convolution layer
    conv_layer += biases

    # if pooling is to be executed. Values taken as default for now. Can keep dict for pooling inside convolution layer dict in future.
    if convolution_properties["pooling"] == "yes":
        conv_layer = tf.nn.max_pool(value=conv_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # activate function gets the activation function corresponding to the enum defined and applies it to the tensor(conv_layer)
    conv_layer = activate(conv_layer, activation_function)

    return conv_layer, weights


# the convolution layer tensor is 4 dimensional. We have to convert it into 2-D. [Number_of_images]*[Image as vector]
def flatten_convolution_layer(conv_layer):
    shape = conv_layer.get_shape()
    # counting the number of elements gives the length of the image vector
    num_features = shape[1:4].num_elements()
    # the flattened layer is of the form [number of images]*[image as vector]. The size of the tensor remains same.
    # in reshape, [-1,..] retains the dimension as in the initial tensor of the number of images.
    # the 2d-image collection for each filter is flattened into 1-d vector
    flattened_layer = tf.reshape(conv_layer, shape=[-1, num_features])

    return flattened_layer, num_features


# creates a fully connected layer
# input : input to the fully connected layer. Should be a flat list.
# fc_properties : properties of the fully connected layer as a dict.
# activation_function : activation function of the layer. Enum.
def new_fc_layer(input, fc_properties, activation_function):
    weights = new_weights([fc_properties["number_of_inputs"], fc_properties["number_of_outputs"]],
                          fc_properties["std_dev"])
    biases = new_bias(fc_properties["number_of_outputs"], 1.0)

    fc_layer = tf.matmul(input, weights) + biases
    fc_layer = activate(fc_layer, activation_function)

    return fc_layer, weights


# Place holder to take value input. Array of vectors. Each vector is an image.
x = tf.placeholder(dtype=tf.float32, shape=[None, cnst.image_size * cnst.image_size], name="x")
# The tensorflow library expects the image to be a 4-dim tensor. Reshape operation to the required shape.
x_img = tf.reshape(x, [-1, cnst.image_size, cnst.image_size, cnst.channels])
# Place holder to take class label input. Format is one-hot.
y = tf.placeholder(dtype=tf.float32, shape=[None, cnst.classes], name="y")
# place holder for class from one-hot input format
y_class = tf.argmax(y, dimension=1)

