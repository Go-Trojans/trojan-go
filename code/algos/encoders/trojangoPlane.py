"""
Author : Rupesh Kumar
Date   : June 8th 2020
File   : tojangoPlane.py

Description : Generate Input features.

"""

import numpy as np
from algos import GlobalConfig


"""
Big Idea for the code written in this file:
===========================================

Assuming 5*5 go board size.

Given: given a board state, generate the input feature stacks/planes (in this case 7*5*5)
Plane[0] = current player (if Black turn then all ones, if white turn then all zeros)

Plane[1] = current player all moves made till now (say ith move) with all ones & others as zeros    (self 1st history)
Plane[2] = current player all moves made till (i-1)th move with all ones & others as zeros          (self 2nd history)
Plane[3] = current player all moves made till (i-2)th move with all ones & others as zeros          (self 3rd history)

Plane[4] = opposotion player all moves made till now (say jth move) with all ones & others as zeros (opp 1st history)
Plane[5] = opposotion player all moves made till (j-1)th move with all ones & others as zeros       (opp 2nd history)
Plane[6] = opposotion player all moves made till (j-2)th move with all ones & others as zeros       (opp 3rd history)

NOTE: if (i-n)th or (j-n)th move doesn't exist then the plane will have all zeros.
"""

"""
Feature name            num of planes   Description
Current Player          1               Player stone / opponent stone / empty
Current 8 history       8               Cuurent Player 8 history states
Opposition 8 history    8               Opposition player 8 history states
"""


FEATURE_OFFSETS = {
    "current_player": 0,    # <1>
    "base_self_history": 1, # <2>
    "base_opp_history": 1 + int((num_planes-1)/2) # <3>
}

# <1> Plane[0]
# <2> Plane [1,2,3,4,5,6,7,8] or [1,2,3]
# <3> Plane [9, 10, 11, 12, 13, 14, 15, 16] or [4,5,6] (num_planes is coming from global config file)

def offset(feature):
    return FEATURE_OFFSETS[feature]




class TrojanGoPlane(Encoder):
        def __init__(self, board_size, plane_size):
            self.board_width, self.board_height = board_size
            self.num_planes = plane_size
	    
        def name(self):
            return 'trojangoplane'

        # Need to define Point, Player, game_state (previous game_state info, 1 black, 2 white and 0 for blank point)
        def encode(self, game_state):    # <1> 
            board_tensor = np.zeros(self.shape())  # (17*19*19) (7*5*5)

            plane_history = 1
            opp = True
            myself = False
            iter_base_opp = 0
            iter_base_self = 0

            if game_state.current_player == Player.black:
                board_tensor[offset("current_player")] = np.ones([1, self.board_width, self.board_width])
            if game_state.current_player == Player.white:
                board_tensor[offset("current_player")] = np.zeros([1, self.board_width, self.board_width])                  


            current_player = game_state.current_player
            while game_state and plane_history <= (num_planes - 1):
                if game_state is None:
                    raise ValueError("encoding history must have neen done by this time ...")
                
                if (opp):
                    # from current player point of view, current game_state is first history
                    # game_state of opposition. So, it should go in opposition base plane.
                    # 2->1 & 1->0 (if game_state.current_player == Player.black),
                    # and 2->0(if game_state.current_player == Player.white)
                    
                    if current_player == Player.black:
                       board_plane = convert2to1and1to0(game_state)
                    else:
                       board_plane = convert2to0(game_state)                 
                    

                    board_tensor[offset("base_opp_history") + iter_base_opp] = board_plane
                    plane_history += 1
                    iter_base_opp += 1
                    opp = False
                    myself = True
                    game_state = game_state.previous
                            
                elif (myself):
                    # 2->0 (if game_state.current_player == Player.black),
                    # and 2->1 & 1->0 (if game_state.current_player == Player.white)
                    
                    if current_player == Player.black:
                       board_plane = convert2to0(game_state)
                    else:
                       board_plane = convert2to1and1to0(game_state)
                       
                    board_tensor[offset("base_self_history") + iter_base_self] = board_plane
                    plane_history += 1
                    iter_base_self+= 1
                    opp = True
                    myself = False
                    game_state = game_state.previous

                else:
                    raise ValueError("Invalid encoding landing ...")
                     
            
        def encode_point(self, point):
            raise NotImplementedError()

        def num_points(self):
            return self.board_width * self.board_height

        def shape(self):
            return self.num_planes, self.board_height, self.board_width

# <1> Encode the input feature (board_size * board_size * num_planes)

def create(board_size, num_planes):
    return TrojanGoPlane(board_size, num_planes)

                
