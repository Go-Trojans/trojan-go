import numpy as np
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
    
