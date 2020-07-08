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
from keras.optimizers import SGD
#from encoders.trojangoPlane import TrojanGoPlane

import numpy as np
import time

class smallNN:
    def __init__(self):
        #self.board_input = Input(shape=TrojanGoPlane.shape(), name='board_input')
        self.board_input = Input(shape=(7, 5, 5), name='board_input')

    def nn_model(self):

        pb = self.board_input
        for i in range(4):                     # <1>
            pb = Conv2D(64, (3, 3),        # <1>
                padding='same',                # <1>
                data_format='channels_first',  # <1>
                activation='relu')(pb)     # <1>



        policy_conv = \
            Conv2D(2, (1, 1),                          # <2>
                data_format='channels_first',          # <2>
                activation='relu')(pb)                 # <2>
        policy_conv_bn = BatchNormalization()(policy_conv)
        policy_flat = Flatten()(policy_conv_bn)           # <2>
        policy_output = \
            Dense(26,
                  activation='softmax')(policy_flat)   # <2>




        value_conv = \
            Conv2D(1, (1, 1),                                    # <3>
                data_format='channels_first',                    # <3>
                activation='relu')(pb)                           # <3>
        value_conv_bn = BatchNormalization()(value_conv)
        value_flat = Flatten()(value_conv_bn)                       # <3>
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



"""
Driver code
"""
if __name__ == "__main__":
    net = smallNN()
    model = net.nn_model()
    print(model.summary())

    model_input = []

    for _ in range(100):
        board_tensor = np.random.randint(0, 3, size=(7, 5, 5))
        model_input.append(board_tensor)

    model_input = np.array(model_input) 
     

    action_target = []
    for _ in range (100):
        search_prob = np.random.randn(26)
        #search_prob_flat = search_prob.reshape(25,)
        action_target.append(search_prob)
        
    action_target = np.array(action_target)    


    value_target = np.random.rand(100)
    value_target = np.array(value_target)

    
    model.compile(SGD(lr=0.01), loss=['categorical_crossentropy', 'mse'])

    start = time.time()
    model.fit(model_input, [action_target, value_target], batch_size=64, epochs=1)
    finish = time.time()
    print("Time taken : ", finish - start)

    X = model_input[0]
    X = np.expand_dims(X, axis=0)
    print(X.shape)
    prediction = model.predict(X)
    print(prediction)

    index = np.argmax(prediction[0])
    rows = int(index/5)
    cols = index%5
    print("Move : ", (rows, cols))
    print("Win chance :", prediction[1])

    

"""
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
board_input (InputLayer)        (None, 7, 5, 5)      0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 64, 5, 5)     4096        board_input[0][0]                
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 64, 5, 5)     36928       conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 64, 5, 5)     36928       conv2d_2[0][0]                   
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 64, 5, 5)     36928       conv2d_3[0][0]                   
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 1, 5, 5)      65          conv2d_4[0][0]                   
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 2, 5, 5)      130         conv2d_4[0][0]                   
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 25)           0           conv2d_6[0][0]                   
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 50)           0           conv2d_5[0][0]                   
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 256)          6656        flatten_2[0][0]                  
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 26)           1326        flatten_1[0][0]                  
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 1)            257         dense_2[0][0]                    
==================================================================================================
Total params: 123,314
Trainable params: 123,314
Non-trainable params: 0
__________________________________________________________________________________________________
None
"""    
