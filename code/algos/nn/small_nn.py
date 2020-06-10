"""
Author : Rupesh Kumar
Date   : June 10th 2020
File   : small_nn.py

Description : Create deep neural network (using CNN and Dense, no ResNet)

Mainly suited for small board size but can be used for any baord size.

"""
# Using Keras Functional Models

from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model
from encoders.trojangoPlane import TrojanGoPlane

class smallNN:
    def __init__(self):
        self.board_input = Input(shape=TrojanGoPlane.shape(), name='board_input')

    def model(self):

        common = self.board_input
        for i in range(4):                     # <1>
            common = Conv2D(64, (3, 3),        # <1>
                padding='same',                # <1>
                data_format='channels_first',  # <1>
                activation='relu')(common)     # <1>



        policy_conv = \                                # <2>
            Conv2D(2, (1, 1),                          # <2>
                data_format='channels_first',          # <2>
                activation='relu')(pb)                 # <2>
        policy_flat = Flatten()(policy_conv)           # <2>
        policy_output = \                              # <2>
            Dense(TrojanGoPlane.num_points() + 1,
                  activation='softmax')(policy_flat)   # <2>




        value_conv = \                                           # <3>
            Conv2D(1, (1, 1),                                    # <3>
                data_format='channels_first',                    # <3>
                activation='relu')(pb)                           # <3>
        value_flat = Flatten()(value_conv)                       # <3>
        value_hidden = Dense(256, activation='relu')(value_flat) # <3>
        value_output = Dense(1, activation='tanh')(value_hidden) # <3>

        model = Model(
            inputs=[self.board_input],
            outputs=[policy_output, value_output])

        return model

# <1> Build a network with 4 convolutional layers. To build a strong bot, you can add many more layers.
# <2> Policy Head (using 2 more layers {1 CNN, 1 Dense} on top of common network).
#     Total output is TrojanGoPlane.num_points() + 1 ( adding one for 'pass' move )
# <3> Value Head  (using 3 more layers {1 CNN, 2 DENSE} on top of common network)

