"""
Author : Puranjay Rajvanshi
Date   : June 9th 2020
File   : game_winner.py
Description : Decides the game winner.
"""

"""
Big Idea for the code written in this file:
===========================================
Assuming any go board size.
Given: given a board state, determine who's the winner
       0 : blank site
       1 : black piece
       2 : White piece 
These are the rules to calculate the winner:
    First all the dead stones of both sides are removed from the board. Then one side's living stones are counted,
    including the vacant points enclosed by those stones. Vacant points situated between both sides' living stones 
    are shared equally. A vacant point counts as one stone.
    The winner is determined by comparison with 180-1/2, which is half the number of points on the board. If the total
    of one side's living stones and enclosed vacant points is larger than 180-1/2, then that side is the winner. If the 
    total is less than 180-1/2, then that side loses. If the total is equal to 180-1/2, the game is a draw.
    In games with compensation, the comparison is made with different numbers, according to separate rules.

Removal of dead stones will be a separate function since it'll be needed for both Suicide an KO functions
"""

import numpy as np
def game_winner(board):
    """

    :param board: A 2-Dimensional numpy array
    :return: 0: Draw
             1: Black wins
             2: White wins

    Basic intuition:
        if black piece is encountered increase points for black by 1
        if white piece is encountered increase points for white by 1
        if blank sites is encountered we try to find the size of the cluster of blank sites
            if blank sites are completely surrounded by black then points for black increases by the group size
            if blank sites are completely surrounded by white then points for white increases by the group size
            if blank sites are completely surrounded by both then points for both increases by (group size)/2
    """
    visited = set()
    m = board.shape[0]
    n = board.shape[1]
    if m == 19:
        komi = 3.75
    else:
        komi = (m/2) - 1
    count_black = -komi
    count_white = komi
    offset = np.array([[1,0],[0,1],[-1,0],[0,-1]])
    for i in range(m):
        for j in range(n):
            if (i,j) in visited:
                continue
            elif board[i][j] == 1:
                count_black+=1
            elif board[i][j] == 2:
                count_white+=1
            elif board[i][j] == 0:
                queue = set()
                queue.add((i,j))
                black_neighbour = False
                white_neighbour = False
                group_count = 0
                while queue:
                    node_x,node_y = queue.pop()
                    if (node_x,node_y) in visited:
                        continue
                    visited.add((node_x,node_y))
                    group_count+=1
                    neighbours = offset+np.array([node_x,node_y])
                    for neighbour in neighbours:
                        if (neighbour[0],neighbour[1]) in visited:
                            continue
                        elif 0<=neighbour[0]<m and 0<=neighbour[1]<n:
                            val = board[neighbour[0]][neighbour[1]]
                            if val == 1:
                                black_neighbour = True
                            elif val == 2:
                                white_neighbour = True
                            elif val == 0:
                                queue.add((neighbour[0],neighbour[1]))
                if black_neighbour and white_neighbour:
                    count_black+=(group_count/2)
                    count_white+=(group_count/2)
                elif black_neighbour:
                    count_black+=group_count
                elif white_neighbour:
                    count_white+=group_count
    if count_white>count_black:
        return 2
    elif count_black>count_white:
        return 1
    else:
        return 0

if __name__ == "__main__":
    board = np.zeros((19,19))
    board[15][8] = 1
    board[5][6] = 2
    print(board)
    print(game_winner(board))