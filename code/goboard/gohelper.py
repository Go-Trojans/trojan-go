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


"""
Driver code
"""
if __name__ == "__main__":

    """
    usage of Player class. Output is mentioned after # for each print statement, so use accordingly.
    """
    print("Player : ", Player.black, Player.black.value) #Player :  Player.black 1
    print("Player : ", Player.white, Player.white.value) #Player :  Player.white 2


    self_player = Player.black
    print(self_player, self_player.value) #Player.black 1

    opp_player = self_player.opp
    print(opp_player, opp_player.value) #Player.white 2
    

    """
    usage of Point
    """
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
    
    



    
    
