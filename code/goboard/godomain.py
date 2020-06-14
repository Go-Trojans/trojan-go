"""
Author :  
Date   : June 11th 2020
File   : goboard.py

Description : Define Go Domain Rules.

Gamestate class is using GoBoard class.

"""

import copy
import numpy as np
from gohelper import Player, Point

__all__ = [
    'GoBoard',
    'GameState',
    'Move',
]


class GoBoard:
    def __init__(self, board_width, board_height, moves = 0):
        self.board_width = board_width
        self.board_height = board_width
        self.grid = np.zeros((board_width, board_width))
        self.komi = 0                                  # <2>
        self.verbose = True                            # <3>
        self.max_move = board_width * board_height * 2 # <4>

# <1> Number of moves played till now.
# <2> placeholder of komi, will see it later on how to use       
# <3> Keeping for debugging purpose, just a knob for enabling/disabling verbose logs.
# <4> The max moves allowed for a Go game,  Games terminate when both players pass or after 19 × 19 × 2 = 722 moves.

    def copy_board(self) :

        return copy.deepcopy(self)

    def place_stone(self, player, point):             # <1>        
        raise NotImplementedError()

# <1> put the stone on the board and take care of other DS like removing dead stone, etc    

    def is_on_grid(self, point):
        return 1 <= point.row <= self.board_width and \
            1 <= point.col <= self.board_height

    def compare_board(self,board) :

        compare = self.grid == board.grid
        return compare.all()

    def display_board(self):
        print(self.grid)

#________________________________________________________________________________________________________________        

#GoState mentioned in the mail but using GameState as class name.
class GameState:
    def __init__(self, board, next_player, previous, last_move,moves=0):
        self.board = board                 # <1>
        self.next_player = next_player     # <2> 
        self.previous_state = previous     # <3>
        self.last_move = last_move         # <4>
        self.moves = moves                 # <5>


# <1> board          : What is the current board
# <2> next_player    : Who's going to make next move.
# <3> previous_state : What was the previous GameState. Or can be referred as Parent.
# <4> last_move      : Last move played (Move.point)


    def apply_move(self, move):
        """Return the new GameState after applying the move."""
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board
        self.update_total_moves()
        return GameState(next_board, self.next_player.other, self, move,self.moves)

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = GoBoard(*board_size)
        return GameState(board, Player.black, None, None)

    def is_suicide(self, player, move):
        raise NotImplementedError()

    def violate_ko(self, player, move):

        test_board = self.board.copy_board()
        test_board.place_stone(player,move.point)

        prev_state = self
        for i in range(8) :
            prev_state = prev_state.previous_state
            if test_board.compare_board(prev_state.board) :
                return True

        return False


    # Return all legal moves
    def legal_moves(self):

        leg_moves = []
        board = self.board
        for r in board.board_height :
            for c in board.board_width :
                move = Move(point=Point(row=r,col=c))
                if self.is_valid_move(move) :
                    leg_moves.append(move)

        return leg_moves

    def is_valid_move(self, move):

        point = move.point
        r,c = point
        board = self.board

        #check if on grid
        if not board.is_on_grid(point) :
            return False
        #check if empty position
        if board.grid[r][c] != 0 :
            return False
        #check no suicide
        if self.is_suicide(self.next_player,move) :
            return False
        #check no KO
        if self.violate_ko(self.next_player,move) :
            return False

        return True

    def update_total_moves(self):
        self.moves = self.moves + 1

    def is_over(self):
        raise NotImplementedError()

    def winner(self):
        raise NotImplementedError()

    
class Move:
    def __init__(self, point=None, is_pass=False):
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_selected = False

    @classmethod
    def play(cls, point):
        return Move(point=point)

    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)
    
    @classmethod
    def ply_selected(cls):
        return Move(is_selected = True)


"""
Driver code
"""
if __name__ == "__main__":
    BOARD_SIZE = 5
    
    """
    usage of Move class
    """
    # Move is (1,1)
    move = Move.play(Point(row=1, col=1))
    print(move.point, move.is_play, move.is_pass, move.is_selected) # Point(row=1, col=1) True False False

    # Move is pass
    move = Move.pass_turn()
    print(move.point, move.is_play, move.is_pass, move.is_selected) # None False True False



    """GameState new_game"""
    gamestate = GameState.new_game(BOARD_SIZE)
    print(gamestate.board.display_board()) # display the board for this gamestate

    # Set player as black
    gamestate.next_player = Player(Player.black.value)
    print(gamestate.next_player)   # Player.black

    """
    bot1 = RandomAgent(Player.black)
    bot2 = RandomAgent(Player.white)

    # Neural network to select the move or let's say you want to play (3,3)
    move = bot1.select_move(gamestate) or move = Move.play(Point(3,3))
    gamestate = gamestate.apply_move(move)
    
    """

    
