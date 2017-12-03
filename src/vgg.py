import numpy as np
import scipy.io
import tensorflow as tf

class VGG:
    def __init__(self, weights_path):
        self.weights = scipy.io.loadmat(weights_path)["layers"][0]

    def _conv_layer(self, input, kernel, bias):
        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding="SAME")
        return tf.nn.bias_add(conv, bias)

    def _max_pool_layer(self, input):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding="SAME")

    def _relu_layer(self, input):
        return tf.nn.relu(input)

    def create_model(self, input, scope):
        layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                  'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                  'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                  'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                  'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4']

        network = {"input": input}
        with tf.variable_scope(scope):
            for index, layer in enumerate(layers):
                layer_kind = layer[:4]
                if layer_kind == "conv":
                    input = self._conv_layer(input, np.transpose(self.weights[index][0][0][2][0][0], (1, 0, 2, 3)),
                                        np.reshape(self.weights[index][0][0][2][0][1], -1))
                elif layer_kind == "relu":
                    input = self._relu_layer(input)
                else:
                    input = self._max_pool_layer(input)

                network[layer] = input

        return network
