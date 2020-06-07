"""
Please keep the code here for the input features.

"""

import numpy as np



"""
Feature name            num of planes   Description
Current Player          1               Player stone / opponent stone / empty
Current 8 history       8               Cuurent Player 8 history states
Opposition 8 history    8               Opposition player 8 history states
"""


FEATURE_OFFSETS = {
    "current_player": 0,   # Plane[0]
    "base_self_history": 1, # Plane [1,2,3,4,5,6,7,8]
    "base_opp_history": 1 + int((plane_size-1)/2)     # Plane [9, 10, 11, 12, 13, 14, 15, 16] (plane_size is coming from global config file)
}


def offset(feature):
    return FEATURE_OFFSETS[feature]




class TrojanGoPlane(Encoder):
        def __init__(self, board_size, plane_size):
	    self.board_width, self.board_height = board_size
	    self.num_planes = plane_size
	    
        def name(self):
            return 'trojangoplane'

        def encode(self, game_state):    # <1> # Need to define game_state (previous game_state info, 1 black, 0 white and -1 for blank point)
            board_tensor = np.zeros(self.shape())  # (17*19*19)

           """
            
            plane_history = 1
            opp = True
            self = False
            iter_base_opp = 0
            iter_base_self = 0
            
            while game_state and plane_history <= (num_planes - 1):
       
                    
                    if (opp):
                            # from current player point of view, current game_state is first history game_state of opposition. So, it should go in opposition base plane.
                            board_tensor[offset("base_opp_history") + iter_base_opp] = game_state # This is wrong. 0->1(change), 1->0(change), -1->0(change)
                            plane_history += 1
                            iter_base_opp += 1
                            opp = False
                            self = True
                            game_state = game_state.previous
                    if (self):
                            board_tensor[offset("base_self_history") + iter_base_self] = game_state # 1->1(no change), 0->0(no change) and -1->0(change)
                            plane_history += 1
                            iter_base_self+= 1
                            opp = True
                            self = False
                            game_state = game_state.previous                 
                    
            """
           
            if game_state.current_player == Player.black: # Need to define Player
                    board_tensor[offset("current_player")] = np.ones([self.board_width, self.board_width, 1])
            if game_state.current_player == Player.white:
                    board_tensor[offset("current_player")] = np.zeros([self.board_width, self.board_width, 1])                  
            

        def encode_point(self, point):
            raise NotImplementedError()

        def num_points(self):
            return self.board_width * self.board_height

        def shape(self):
            return self.num_planes, self.board_height, self.board_width

# <1> Encode the input feature (board_size * board_size * num_planes)

def create(board_size, num_planes):
    return TrojanGoPlane(board_size, num_planes)

                
