"""
Author : Rupesh Kumar
Date   : June 11th 2020
File   : tojangoPlane.py

Description : Define Player and Point class.

"""

import enum
from collections import namedtuple
import operator

__all__ = [
    'Player',
    'Point',
    'is_point_an_eye',
]

def directionVector_add(a, b):
    return tuple(map(operator.add, a, b))

directions = {'W': (-1, 0), 'E': (1, 0), 'S': (0, -1), 'N': (0, 1)}
action = {1 : 'W', 2 : 'E', 3 : 'S', 4 : 'N'}


class Player(enum.Enum):
    black = 1
    white = 2

    @property
    def opp(self):  #opposition
        if self == Player.white:
            return Player.black 
        else:
            return Player.white

class Point(namedtuple('Point', 'row col')):
    def neighbors(self):
        neigh = []
        for i in range (1, len(directions) + 1):
            row, col = directionVector_add((self.row, self.col),(directions[action[i]]))
            neigh.append(Point(row, col))
            
        return neigh

    def __deepcopy__(self, memodict={}):
        return self



def is_point_an_eye(board, point, player):
    """
    Checks if the point is an eye for the player being passed in (not evaluating if point is an eye for either player). 
    Checks if point is:
        1) a liberty
        2) surrounded by a chain or almost chain
            a) neighbors are the same color
            b) neighbors are part of the same chain
    """
    r, c = point
    if board.grid[r][c] != 0:
        return False

    neighs = point.neighbors()

    chains = set()
    visited = set()
    adj_libs = set()
            
    # neighbors are the same color and are part of a chain
    for n in neighs: 
        nr, nc = n

        # if neighbor is invalid then we skip computations
        if nr < 0 or nc < 0 or not board.is_on_grid(Point(row=nr, col=nc)):
            continue

        # if a neighbor isn't occupied by current player, then it's not an eye
        if board.grid[nr][nc] != player.value:
            return False

        if n not in visited:
            curr_chain, adj_lib = find_connected(board, n, player)
            visited = visited.union(curr_chain)
            if len(adj_libs) > 0:
                adj_libs = adj_libs.intersection(adj_lib)
            else:
                adj_libs = adj_lib
            chains.add(tuple(curr_chain))

    if len(chains) == 1:
        return True

    # print(chains)
    # print(adj_libs)
    lib_count = 0
    for lib in adj_libs:
        if lib == point:
            continue
        # print(lib)
        board.grid[lib[0]][lib[1]] = player.value
        c, _ = find_connected(board, lib, player)
        ss_count = 0
        for chain in chains:
            # print("c %s" % c)
            # print("chain %s" % str(chain))
            # print(set(chain).issubset(c))
            if set(chain).issubset(c):
                ss_count += 1
            else:
                break
        # print(ss_count)
        if ss_count == len(chains):
            # print(lib)
            lib_count += 1
        board.grid[lib[0]][lib[1]] = 0
    # print(lib_count)
    if lib_count > 1:
        return True
    return False
    
def find_connected(board, point, player):
    """
    BFS from starting point to get all pieces in a connected group. 
    """
    #print("point, player : ", point, player)
    queue = [point]

    visited = set() # you're only visiting neighbors in same chain
    adj_lib = set()
    while len(queue) > 0:
        curr = queue.pop(0)
        neighs = curr.neighbors()
        visited.add(curr)
        for n in neighs:
            r, c = n
            if not board.is_on_grid(point) or not board.is_on_grid(Point(row=r, col=c)):
                continue
            if n not in visited and board.grid[r][c] == player.value:
                queue.append(n)
            if board.grid[r][c] == 0:
                adj_lib.add(n)
    return visited, adj_lib

"""
#Driver code
if __name__ == "__main__":

    #usage of Player class. Output is mentioned after # for each print statement, so use accordingly.
    print("Player : ", Player.black, Player.black.value) #Player :  Player.black 1
    print("Player : ", Player.white, Player.white.value) #Player :  Player.white 2


    self_player = Player.black
    print(self_player, self_player.value) #Player.black 1

    opp_player = self_player.opp
    print(opp_player, opp_player.value) #Player.white 2
    

    #usage of Point
    print("Point is : ", Point(row=1, col=1)) # Point is :  Point(row=1, col=1)

    row = 2
    col = 2
    print("Point is : ", Point(row, col)) # Point is :  Point(row=2, col=2)

    point = Point(row=3, col=3)
    print(point) # Point(row=3, col=3)

    #Find the neighbors of Point (row=3,col=3). Any point have 4 neighbors
    neighs = point.neighbors()
    for neigh in neighs:
        print(neigh) # Point(row=2, col=3) Point(row=4, col=3) Point(row=3, col=2) Point(row=3, col=4)
    
"""
