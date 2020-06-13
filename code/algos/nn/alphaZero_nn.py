"""
Author : 
File   : alphaZero_nn.py
Date   : 11th July 2020

Description : Have the same deep neural network as AlphaGoZero
"""


"""
s{t} = [X{t}, Y{t}, X{t−1}, Y{t−1},..., X{t−7}, Y{t−7}, C].
The input features s{t} are processed by a residual tower that consists of a single convolutional block 
followed by either 19 or 39 residual blocks.

The convolutional block applies the following modules:
(1) A convolution of 256 filters of kernel size 3 × 3 with stride 1
(2) Batch normalization
(3) A rectifier nonlinearity

Each residual block applies the following modules sequentially to its input: 
(1) A convolution of 256 filters of kernel size 3 × 3 with stride 1
(2) Batch normalization
(3) A rectifier nonlinearity
(4) A convolution of 256 filters of kernel size 3 × 3 with stride 1
(5) Batch normalization
(6) A skip connection that adds the input to the block
(7) A rectifier nonlinearity

The output of the residual tower is passed into two separate ‘heads’ for computing the policy and value. 

The policy head applies the following modules: 
(1) A convolution of 2 filters of kernel size 1 × 1 with stride 1
(2) Batch normalization
(3) A rectifier nonlinearity
(4) A fully connected linear layer that outputs a vector of size 19*19 + 1 = 362, corresponding to logit probabilities for all intersections and the pass move

The value head applies the following modules:
(1) A convolution of 1 filter of kernel size 1 × 1 with stride 1 
(2) Batch normalization
(3) A rectifier nonlinearity
(4) A fully connected linear layer to a hidden layer of size 256
(5) A rectifier nonlinearity
(6) A fully connected linear layer to a scalar
(7) A tanh nonlinearity outputting a scalar in the range [−1, 1]

The overall network depth, in the 20­ or 40­block network, is 39 or 79 parameterized layers, respectively, 
for the residual tower, plus an additional 2 layers for the policy head and 3 layers for the value head.
"""

from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model


class trojanGoZero:
    def __init__(self):
        self.board_input = Input(shape=TrojanGoPlane.shape(), name='board_input')

    def nn_model(self):
        #return model
        raise NotImplementedError() 
    
    
