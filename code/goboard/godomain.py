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
import remove_dead_stones
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

    def __eq__(self, other) :
        if isinstance(other, GoBoard) :
            comparison = self.grid == other.grid
            return comparison.all()
        return False
    def copy_board(self) :

        return copy.deepcopy(self)

    def place_stone(self, player, point):             # <1>        
        r, c = point
        self.grid[r][c] = player.value
        remove_dead_stones.remove_dead_stones(self.grid, player.value)

# <1> put the stone on the board and take care of other DS like removing dead stone, etc    

    def is_on_grid(self, point):
        return 0 <= point.row <= self.board_width and \
            0 <= point.col <= self.board_height

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

    def detect_neighbor_ally(self,player,point):

        grid = self.board.grid
        neighbors = point.neighbors()  # Detect neighbors
        group_allies = []

        for piece in neighbors:
            if self.board.is_on_grid(piece) :
                nR,nC = piece
                if grid[nR][nC] == player.value:
                    group_allies.append(piece)
        return group_allies

    def ally_dfs(self, player, point):

        stack = [point]
        ally_members = []
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(player,point)
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def is_suicide(self,player,move):

        grid = self.board.grid
        point = move.point
        ally_members = self.ally_dfs(player,point)
        for member in ally_members:
            neighbors = member.neighbors()
            for piece in neighbors:
                if self.board.is_on_grid(piece) :
                    nR,nC = piece
                    # If there is empty space around a piece, it has liberty
                    if grid[nR][nC] == 0:
                        return False
        return True

    def violate_ko(self, player, move):

        test_board = self.board.copy_board()
        test_board.place_stone(player,move.point)

        prev_state = self
        for i in range(8) :
            prev_state = prev_state.previous_state
            if not prev_state :
                break
            if test_board == prev_state.board :
                return True

        return False

    # Return all legal moves
    def legal_moves(self) :
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

        #check if off grid or not empty position
        if not board.is_on_grid(point) or board.grid[r][c] != 0 :
            return False
        #check KO or suicide
        if self.violate_ko(self.next_player,move) or self.is_suicide(self.next_player,move):
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

    
