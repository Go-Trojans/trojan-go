#!/usr/bin/env python
# coding: utf-8

# # s{t} = [X{t}, Y{t}, X{t−1}, Y{t−1},..., X{t−7}, Y{t−7}, C].
# The input features s{t} are processed by a residual tower that consists of a single convolutional block 
# followed by either 19 or 39 residual blocks.
# 
# The convolutional block applies the following modules:
# (1) A convolution of 256 filters of kernel size 3 × 3 with stride 1
# (2) Batch normalization
# (3) A rectifier nonlinearity
# 
# Each residual block applies the following modules sequentially to its input: 
# (1) A convolution of 256 filters of kernel size 3 × 3 with stride 1
# (2) Batch normalization
# (3) A rectifier nonlinearity
# (4) A convolution of 256 filters of kernel size 3 × 3 with stride 1
# (5) Batch normalization
# (6) A skip connection that adds the input to the block
# (7) A rectifier nonlinearity
# 
# The output of the residual tower is passed into two separate ‘heads’ for computing the policy and value. 
# 
# The policy head applies the following modules: 
# (1) A convolution of 2 filters of kernel size 1 × 1 with stride 1
# (2) Batch normalization
# (3) A rectifier nonlinearity
# (4) A fully connected linear layer that outputs a vector of size 19*19 + 1 = 362, corresponding to logit probabilities for all intersections and the pass move
# 
# The value head applies the following modules:
# (1) A convolution of 1 filter of kernel size 1 × 1 with stride 1 
# (2) Batch normalization
# (3) A rectifier nonlinearity
# (4) A fully connected linear layer to a hidden layer of size 256
# (5) A rectifier nonlinearity
# (6) A fully connected linear layer to a scalar
# (7) A tanh nonlinearity outputting a scalar in the range [−1, 1]
# 
# The overall network depth, in the 20­ or 40­ block network, is 39 or 79 parameterized layers, respectively, 
# for the residual tower, plus an additional 2 layers for the policy head and 3 layers for the value head.
# """
# 

# In[105]:


import keras
from keras.layers import Activation, BatchNormalization,MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model
import numpy as np
import tensorflow as tf

from algos.godomain import Move
from algos.gohelper import Point, Player

""" return True if tensorflow version is equal or higher than 2.0, else False """
def tf_version_comp(tf_v):
    tf_v = tf_v.split(".")
    if int(tf_v[0]) == 2 and int(tf_v[1]) >=0:
        return True
    return False
    
"""
We faced a problem when we have a GPU computer that shared with multiple users.
Most users run their GPU process without the “allow_growth” option in their Tensorflow or Keras environments.
It causes the memory of a graphics card will be fully allocated to that process.
In reality, it is might need only the fraction of memory for operating.
It prevents any new GPU process which consumes a GPU memory to be run on the same machine.
"""        
if tf_version_comp(tf.__version__):
    config = tf.compat.v1.ConfigProto() 
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
else:
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)


"""
This is 4 layers ResNet network.
"""
class smallNN:
    def __init__(self, model=None):
        # self.board_input = Input(shape=TrojanGoPlane.shape(), name='board_input')
        self.board_input = Input(shape=(7, 5, 5), name='board_input')
        self.model = model

    def nn_model(self, input_shape):
        pb = self.board_input
        for i in range(4):  # <1>
            pb = Conv2D(64, (3, 3),  # <1>
                        padding='same',  # <1>
                        data_format='channels_first',  # <1>
                        activation='relu')(pb)  # <1>

        policy_conv = \
            Conv2D(2, (1, 1),  # <2>
                   data_format='channels_first',  # <2>
                   activation='relu')(pb)  # <2>
        policy_conv_bn = BatchNormalization()(policy_conv)
        policy_flat = Flatten()(policy_conv_bn)  # <2>
        policy_output = \
            Dense(26,
                  activation='softmax')(policy_flat)  # <2>

        value_conv = \
            Conv2D(1, (1, 1),  # <3>
                   data_format='channels_first',  # <3>
                   activation='relu')(pb)  # <3>
        value_conv_bn = BatchNormalization()(value_conv)
        value_flat = Flatten()(value_conv_bn)  # <3>
        value_hidden = Dense(256, activation='relu')(value_flat)  # <3>
        value_output = Dense(1, activation='tanh')(value_hidden)  # <3>

        model = Model(
            inputs=[self.board_input],
            outputs=[policy_output, value_output])

        return model

    def select_move(self, encoder, gameState):
        tensor = encoder.encode(gameState)
        tensor = np.expand_dims(tensor,axis=0)
        p,v = self.model.predict(tensor) # self.mdel should be fine I guess.

        """ argmax is not necessarily the best move each time as
            it would have played already and so it could be invalid move.
            So, arrange the prediction(p) in decreasing order of
            probability value and select the best move.
        """
        sorted_indexes = np.argsort(p.tolist()[0])[::-1]
        for index in sorted_indexes:
            rows = int(index / encoder.board_height)
            cols = int(index % encoder.board_width)

            # Move is pass , i.e. index = 25
            if (rows == encoder.board_height and cols == 0):
                move = Move.pass_turn()
                #print(gameState.next_player, move.point)
                return move
            else:
                move = Move.play(Point(row=rows, col=cols))
                #print(gameState.next_player, move.point)
                #check the move is valid or not
                if gameState.is_valid_move(move):
                    return move
                
        #code should not reach here
        print("[ERROR] NN SELECT_MOVE RETURNING None")        
        return None

        
        

"""
This is 6 layers ResNet network.
"""
class mediumNN:
    def __init__(self):
        # self.board_input = Input(shape=TrojanGoPlane.shape(), name='board_input')
        self.board_input = Input(shape=(7, 5, 5), name='board_input')

    def nn_model(self):
        pb = self.board_input
        for i in range(6):  # <1>
            pb = Conv2D(64, (3, 3),  # <1>
                        padding='same',  # <1>
                        data_format='channels_first',  # <1>
                        activation='relu')(pb)  # <1>

        policy_conv = \
            Conv2D(2, (1, 1),  # <2>
                   data_format='channels_first',  # <2>
                   activation='relu')(pb)  # <2>
        policy_conv_bn = BatchNormalization()(policy_conv)
        policy_flat = Flatten()(policy_conv_bn)  # <2>
        policy_output = \
            Dense(26,
                  activation='softmax')(policy_flat)  # <2>

        value_conv = \
            Conv2D(1, (1, 1),  # <3>
                   data_format='channels_first',  # <3>
                   activation='relu')(pb)  # <3>
        value_conv_bn = BatchNormalization()(value_conv)
        value_flat = Flatten()(value_conv_bn)  # <3>
        value_hidden = Dense(256, activation='relu')(value_flat)  # <3>
        value_output = Dense(1, activation='tanh')(value_hidden)  # <3>

        model = Model(
            inputs=[self.board_input],
            outputs=[policy_output, value_output])

        return model

"""
This is 19 layers (num_resnet_block) ResNet network.
"""
class trojanGoZero:
    def __init__(self, num_resnet_block=19):
        # self.board_input = Input(shape=TrojanGoPlane.shape(), name='board_input')
        self.board_input = Input(shape=(7, 5, 5), name='board_input')
        self.num_resnet_block = num_resnet_block
        self.num_filters = 256

    def resNetBlock(self, x, filters, pool=False):
        res = x

        if pool:
            x = MaxPooling2D(pool_size=(2, 2))(x)
            res = Conv2D(filters=filters, kernel_size=[1, 1], strides=(2, 2), padding="same",
                         data_format='channels_first')(res)
        # out = BatchNormalization()(x)
        # out = Activation("relu")(out)

        out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same", data_format='channels_first')(
            x)
        out = BatchNormalization()(out)
        out = Activation("relu")(out)

        out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same", data_format='channels_first')(
            out)
        out = BatchNormalization()(out)

        out = keras.layers.add([res, out])
        out = Activation("relu")(out)

        return out

    def nn_model(self, input_shape):
        # Input feature of 17*19*19 or 7*5*5 as board_input or board_images

        board_images = Input(input_shape)
        # board_images = self.board_input

        # CNN-1 with Batch Normalization and rectifier nonlinearity.
        cnn1 = Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding="same", data_format='channels_first')(
            board_images)
        cnn1_batch = BatchNormalization()(cnn1)
        cnn1_act = Activation("relu")(cnn1_batch)

        self_in = cnn1_act

        # Now build 19 or 39 ResNet block networks depends on "num_resnet_block" variable.
        for i in range(self.num_resnet_block):
            self_out = self.resNetBlock(self_in, self.num_filters)
            self_in = self_out

        out = self_out

        policy_conv = \
            Conv2D(2, (1, 1),  # <2>
                   data_format='channels_first',  # <2>
                   activation='relu')(out)  # <2>
        policy_conv_bn = BatchNormalization()(policy_conv)
        policy_flat = Flatten()(policy_conv_bn)  # <2>
        policy_output = \
            Dense(26,
                  activation='softmax')(policy_flat)  # <2>

        value_conv = \
            Conv2D(1, (1, 1),  # <3>
                   data_format='channels_first',  # <3>
                   activation='relu')(out)  # <3>
        value_conv_bn = BatchNormalization()(value_conv)
        value_flat = Flatten()(value_conv_bn)  # <3>
        value_hidden = Dense(256, activation='relu')(value_flat)  # <3>
        value_output = Dense(1, activation='tanh')(value_hidden)  # <3>

        model = Model(
            inputs=[board_images],
            outputs=[policy_output, value_output])

        return model


# In[106]:

def init_random_model(input_shape) :
    #net = trojanGoZero()
    net = smallNN()
    model = net.nn_model(input_shape)
    return model


# In[ ]:




