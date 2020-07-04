from tensorflow import keras

"""
File: alphago_zero_nn.py
Author: David Lin

AlphaGo Zero NN keras implementation. 
"""

class NN:

    def __init__(self, channels=7, board_size=5, filters=256):
        self.filters = filters
        self.board_size = board_size
        self.channels = channels

    def _conv_block(self, x):
        """
        The convolutional block applies the following modules: 
        (1) A convolution of 256 filters of kernel size 3 × 3 with stride 1 
        (2) Batch normalization 
        (3) A rectifier nonlinearity
        """
        conv_layer = keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=[1, 1], data_format="channels_last", padding="same")(x)
        bn = keras.layers.BatchNormalization()(conv_layer)
        output = keras.layers.Activation("relu")(bn)
        return output

    def _res_block(self, x):
        """
        Each residual block applies the following modules sequentially to its input: 
        (1) A convolution of 256 filters of kernel size 3 × 3 with stride 1 
        (2) Batch normalization 
        (3) A rectifier nonlinearity 
        (4) A convolution of 256 filters of kernel size 3 × 3 with stride 1 
        (5) Batch normalization 
        (6) A skip connection that adds the input to the block 
        (7) A rectifier nonlinearity
        """
        conv_1 = keras.layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=[1, 1], padding="same", data_format="channels_last")(x)
        bn_1 = keras.layers.BatchNormalization()(conv_1)
        act_1 = keras.layers.Activation("relu")(bn_1)
        conv_2 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=[1, 1], data_format="channels_last", padding="same")(act_1)
        bn_2 = keras.layers.BatchNormalization()(conv_2)

        # skip = keras.layers.Conv2D(filters=self.filters, strides=(1, 1), kernel_size=(1, 1),data_format="channels_last", padding="same")(x)
        merge = keras.layers.add([x, bn_2])
        output = keras.layers.Activation("relu")(merge)
        return output

    def _policy(self, x):
        """
        The policy head applies the following modules: 
        (1) A convolution of 2 filters of kernel size 1 × 1 with stride 1 
        (2) Batch normalization 
        (3) A rectifier nonlinearity 
        (4) A fully connected linear layer that outputs a vector of size 19*19 + 1 = 362, corresponding to logit probabilities for all intersections and the pass move
        """

        conv = keras.layers.Conv2D(filters=2, kernel_size=(1, 1), data_format="channels_last", padding="same")(x)
        # bn = keras.layers.BatchNormalization()(conv)
        act = keras.layers.Activation("relu")(conv)
        flat = keras.layers.Flatten(data_format="channels_last")(act)
        output = keras.layers.Dense(self.board_size ** 2 + 1, activation="softmax")(flat)
        return output

    def _value(self, x):
        """
        The value head applies the following modules: 
        (1) A convolution of 1 filter of kernel size 1 × 1 with stride 1 
        (2) Batch normalization 
        (3) A rectifier nonlinearity 
        (4) A fully connected linear layer to a hidden layer of size 256 
        (5) A rectifier nonlinearity 
        (6) A fully connected linear layer to a scalar 
        (7) A tanh nonlinearity outputting a scalar in the range [−1, 1]
        """

        conv = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same", data_format="channels_last")(x)
        # bn = keras.layers.BatchNormalization()(conv)
        act_1 = keras.layers.Activation("relu")(conv)
        flat = keras.layers.Flatten(data_format="channels_last")(act_1)
        dense_1 = keras.layers.Dense(self.filters)(flat)
        act_2 = keras.layers.Activation("relu")(dense_1)
        output = keras.layers.Dense(1, activation="tanh")(act_2)
        return output

    def create_tower(self, num_res_blocks=19):
        """
        param(s): 
        input_shape - tuple with 3 fields: channels, and board width/height
        num_res_blocks - number of residual blocks we want to use in our model

        return: keras model
        """
        inputs = keras.layers.Input(shape=(self.board_size, self.board_size, self.channels))
        outputs = self._conv_block(inputs)
        for _ in range(num_res_blocks):
            outputs = self._res_block(outputs)
        p, v = self._policy(outputs), self._value(outputs)
        model = keras.models.Model(inputs, [p, v])
        return model

if __name__ == '__main__':
    import tensorflow as tf

    strategy = tf.distribute.MirroredStrategy()
    print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = NN(channels=17, board_size=19).create_tower()
        model.compile(optimizer='sgd', loss=['categorical_crossentropy', 'mse'])
        # print(model.summary())

    import numpy as np
    rand_input = np.random.randint(3, size=(10000, 19, 19, 17))
    print(rand_input.shape)

    action_target = np.random.rand(10000, 362)
    print(action_target.shape)

    value_target = np.random.rand(10000)
    print(value_target.shape)

    import time
    start = time.time()
    model.fit(x=rand_input, y=[action_target, value_target], batch_size=64, epochs=1)
    end = time.time()
    print(end-start)
