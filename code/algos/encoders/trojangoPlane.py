"""
Author : Rupesh Kumar
Date   : June 8th 2020
File   : tojangoPlane.py

Description : Generate Input features.

"""

import numpy as np
from algos import GlobalConfig
from utility import convert2to1and1to0, convert2to0


"""
Big Idea for the code written in this file:
===========================================

Assuming 5*5 go board size.

Given: given a board state, generate the input feature stacks/planes (in this case 7*5*5)
C = Plane[0] = current player (if Black turn then all ones, if white turn then all zeros)

X{t=2} = Plane[1] = current player all moves made till now, say (i)th move with all ones & others as zeros   (self 1st history)
X{t=1} = Plane[2] = current player all moves made till (i-1)th move with all ones & others as zeros          (self 2nd history)
X{t=0} = Plane[3] = current player all moves made till (i-2)th move with all ones & others as zeros          (self 3rd history)

Y{t=2} = Plane[4] = opposotion player all moves made till now (say jth move) with all ones & others as zeros (opp 1st history)
Y{t=1} = Plane[5] = opposotion player all moves made till (j-1)th move with all ones & others as zeros       (opp 2nd history)
Y{t=0} = Plane[6] = opposotion player all moves made till (j-2)th move with all ones & others as zeros       (opp 3rd history)

NOTE: if (i-n)th or (j-n)th move doesn't exist then the plane will have all zeros.
AlphaZero: These planes are concatenated together to give input features s{t} = [X{t}, Y{t}, X{t−1}, Y{t−1},..., X{t−7}, Y{t−7}, C].
board_tensor, s{t} = [C, X{t=2}, X{t=1}, X{t=0}, Y{t=2}, Y{t=1}, Y{t=0}]

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
                
            """
            return board_tensor
            s{t} = [C, X{t=2}, X{t=1}, X{t=0}, Y{t=2}, Y{t=1}, Y{t=0}]
            
            AlphaZero: These planes are concatenated together to give input features
            s{t} = [X{t}, Y{t}, X{t−1}, Y{t−1},..., X{t−7}, Y{t−7}, C].
            So, re-sequence it to align with AlphaGoZero.
            """
            
            new_board_tensor = np.zeros(self.shape())
            new_board_tensor[-1] = board_tensor[0]

            j = (self.num_planes - 1) / 2      # number of history staes for any player
            k = -1
            for i in range(self.num_planes - 1):
                if i%2 == 0:                  # current player planes re-sequencing X{t}
                    new_board_tensor[i] = board_tensor[i+ (-1 * k)]
                    k = k+1
                else:                         # opp player planes re-sequencing Y{t}
                    new_board_tensor[i] = board_tensor[i+j]
                    j = j-1
                    
                
            return new_board_tensor  # AlphaGoZero complaint
                     
            
        def encode_point(self, point):
            raise NotImplementedError()

        def num_points(self):
            return self.board_width * self.board_height

        def shape(self):
            return self.num_planes, self.board_height, self.board_width

# <1> Encode the input feature (board_size * board_size * num_planes)

def create(board_size, num_planes):
    return TrojanGoPlane(board_size, num_planes)


"""
Driver code
"""
if __name__ == "__main__":
    board_size = 5
    num_planes = 7
    
    board = Board(board_size, board_size)
    board.new_game()
    #board.print_board()

    # black is starting the game, board is all empty now.
    game_state = GameState(board, Player.black, None)
    #print(game_state.display())

    # Simulate a game for making 3-3 moves for black and white.
    #moves = [(2,2), (2,3), (2,1), (3,3), (1,1), (1,2), (4,4), (0,0)]
    moves = [(2,2), (2,3), (2,1), (3,3)]
    
    for coord in moves:
        game_state = game_state.playGame(coord)
        
    print(game_state.display())

    #now get an input feature stack
    trojangoplane = create(board_size, num_planes)
    planes_tensor = trojangoplane.encode(game_state)

    print("Turn to make move is player : ", game_state.current_player)
    print("Input feature stacks ...")
    print(planes_tensor)
    
                
