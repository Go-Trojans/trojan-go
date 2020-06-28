"""
Please keep the code which we need for testing here.
"""

import numpy as np
import enum
from collections import namedtuple
from copy import deepcopy



"""
Convert to make the plane for Black (Game State represenation assumed: Black(1), White(2) and empty(0))
"""
def convert2to0(game_state):
    if game_state is None:
        print("game_state is None")
        return None
    
    board = deepcopy(game_state.board)
    white_points = [(i,j) for i in range(5) for j in range(5) if board.grid[i][j] == 2]
    for row,col in white_points:
        board.grid[row][col] = 0
    return board.grid

"""
Convert to make the plane for White (Game State represenation assumed: Black(1), White(2) and empty(0))
"""
def convert2to1and1to0(game_state):
    if game_state is None:
        print("game_state is None")
        return None
    
    board = deepcopy(game_state.board)
    black_points = [(i,j) for i in range(5) for j in range(5) if board.grid[i][j] == 1]
    for row,col in black_points:
        board.grid[row][col] = 0
    white_points = [(i,j) for i in range(5) for j in range(5) if board.grid[i][j] == 2]
    for row,col in white_points:
        board.grid[row][col] = 1        
    return board.grid
    



class Player(enum.Enum):
    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white

class Point(namedtuple('Point', 'row col')):
    def neighbors(self):
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]

    def __deepcopy__(self, memodict={}):
        # These are very immutable.
        return self

class Board:
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.grid = []

    def new_game(self):
        self.grid = np.zeros((self.num_rows, self.num_cols))

    def print_board(self):
        print(self.grid)
        
class GameState:
    def __init__(self, board, current_player, previous):
        self.board = board
        self.current_player = current_player
        self.previous_state = previous

    def playGame(self, coord):
        next_board = deepcopy(self.board)
        row, col = coord
        val = 0
        if self.current_player == Player.black:
            val = 1
        else:
            val = 2
        next_board.grid[int(row)][int(col)] = val
        next_game_state = GameState(next_board, self.current_player.other, self)
        return next_game_state
        

    def display(self):
        print("Current Player : ", self.current_player)
        game_state = self
        while(game_state):
            print("board for player :", game_state.current_player)
            print(game_state.board.grid)
            game_state = game_state.previous_state

num_planes = 7

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




class TrojanGoPlane():
        def __init__(self, board_size, plane_size):
            self.board_width, self.board_height = board_size
            self.num_planes = plane_size
	    
        def name(self):
            return 'trojangoplane'

        # Need to define Point, Player, game_state (previous game_state info, 1 black, 2 white and 0 for blank point)
        def encode(self, game_state):    # <1> 
            board_tensor = np.zeros(self.shape())  # (17*19*19)

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
                    #print("I WAS EXPECTING TO BREAK THE WHILE LOOP, do it now ...")
                    #break
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
                    game_state = game_state.previous_state
                            
                elif (myself):
                    # 2->0 (if game_state.current_player == Player.black)5,
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
                    game_state = game_state.previous_state
                    
                    
                else:
                    raise ValueError("INVALID PLAY LANDING")

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
            j = int(j)
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

def create(board_size, num_planes, gamestate):
    trojangoplane = TrojanGoPlane((board_size, board_size), num_planes)
    return trojangoplane.encode(game_state)


    


if __name__ == "__main__":
    board = Board(5,5)
    board.new_game()
    #board.print_board()

    # black is starting the game, board is all empty now.
    game_state = GameState(board, Player.black, None)
    #print(game_state.display())

    # Simulate a game for making 3-3 moves for black and white.
    #moves = [(2,2), (2,3), (2,1), (3,3), (1,1), (1,2)]
    moves = [(2,2), (2,3), (2,1), (3,3), (1,1), (1,2), (4,4), (0,0)]
    #moves = [(2,2), (2,3), (2,1), (3,3)]
    
    for coord in moves:
        game_state = game_state.playGame(coord)
        
    print(game_state.display())

    #now get an input feature stack
    planes_tensor = create(5, 7, game_state)

    print("Turn to make move is player : ", game_state.current_player)
    print("Input feature stacks ...")
    print(planes_tensor)
    

   
                

