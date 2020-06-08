import numpy as np
from copy import deepcopy

"""
Convert to make the plane for Black (Game State represenation assumed: Black(1), White(2) and empty(0))
"""
def convert2to0(game_state):
    board_plane = deepcopy(game_state)
    white_points = [(i,j) for i in range(5) for j in range(5) if game_state[i][j] == 2]
    for row,col in white_points:
        board_plane[row][col] = 0
    return board_plane

"""
Convert to make the plane for White (Game State represenation assumed: Black(1), White(2) and empty(0))
"""
def convert2to1and1to0(game_state): 
    board_plane = deepcopy(game_state)
    black_points = [(i,j) for i in range(5) for j in range(5) if game_state[i][j] == 1]
    for row,col in black_points:
        board_plane[row][col] = 0
    white_points = [(i,j) for i in range(5) for j in range(5) if game_state[i][j] == 2]
    for row,col in white_points:
        board_plane[row][col] = 1        
    return board_plane
    
