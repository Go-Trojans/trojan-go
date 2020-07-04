"""
Author :  
Date   : June 11th 2020
File   : goboard.py

Description : Define Go Domain Rules.

Gamestate class is using GoBoard class.

"""

import copy
import numpy as np
from algos.gohelper import Player, Point

__all__ = [
    'GoBoard',
    'GameState',
    'Move',
]


class GoBoard:
    def __init__(self, board_width, board_height, moves=0):
        self.board_width = board_width
        self.board_height = board_width
        self.grid = np.zeros((board_width, board_width), dtype=int)
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

    def place_stone(self, player, point):  # <1>
        """
        Assuming that the valid point check has been applied prior to
        calling this function. Just sets the coordinates on grid.
        """
        r, c = point
        #move = Move.play(Point(row=r, col=c))
        
        self.grid[r][c] = player.value
        self.remove_dead_stones(player, np.array([r,c]))

    def remove_dead_stones(self, player, move):
        """

            :param 
                   player: Current player
                   move: The latest move played
            :return: board: A 2-Dimensional numpy array with the dead pieces removed (Not needed)

            Basic intuition:
                find enemy groups neighbouring the latest move with no liberties and remove them from the board.
            """
        board = self.grid # this is not board but just a grid.
        piece = player.value
        
        
        visited = set()
        m = board.shape[0]
        n = board.shape[1]
        piece = 3 - piece
        
        offset = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        move_neighbours = offset + move

        for move_neighbour in move_neighbours:
            r, c = move_neighbour[0], move_neighbour[1]
            point = Point(row=r, col=c)
            if not self.is_on_grid(point):
                continue
            
            if board[move_neighbour[0]][move_neighbour[1]] == piece:
                if (move_neighbour[0], move_neighbour[1]) in visited:
                    continue
                liberty_found = False
                remove_group = []
                queue = set()
                queue.add((move_neighbour[0], move_neighbour[1]))
                while queue:
                    node_x, node_y = queue.pop()
                    if (node_x, node_y) in visited:
                        continue
                    visited.add((node_x, node_y))
                    remove_group.append([node_x, node_y])
                    neighbours = offset + np.array([node_x, node_y])
                    for neighbour in neighbours:
                        if (neighbour[0], neighbour[1]) in visited:
                            continue
                        if 0 <= neighbour[0] < m and 0 <= neighbour[1] < n:
                            val = board[neighbour[0]][neighbour[1]]
                            if val == 0:
                                liberty_found = True
                            if val == piece:
                                queue.add((neighbour[0], neighbour[1]))

                if not liberty_found:
                    while remove_group:
                        del_node_x, del_node_y = remove_group.pop()
                        board[del_node_x][del_node_y] = 0

    def is_on_grid(self, point):
        return 0 <= point.row < self.board_width and \
               0 <= point.col < self.board_height

    def display_board(self):
        print(self.grid)


# ________________________________________________________________________________________________________________

# GoState mentioned in the mail but using GameState as class name.
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
    # <5> moves          : Total moves played so far

    def copy(self) :

        return copy.deepcopy(self)

    def apply_move(self, move):
        """Return the new GameState after applying the move."""

        # if we don't pass
        if move.is_play:
            # If the move is Invalid then print invalid move and return
            if not self.board.is_on_grid(move.point) or not self.is_valid_move(move):
                raise ValueError("Invalid move")
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board
        #self.update_total_moves()
        return GameState(next_board, self.next_player.opp, self, move,self.moves+1)

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

        if not move.is_play:
            return False

        test_state = self.copy()
        grid = test_state.board.grid
        point = move.point
        test_state.board.place_stone(player,point)
        ally_members = test_state.ally_dfs(player,point)
        for member in ally_members:
            neighbors = member.neighbors()
            for piece in neighbors:
                if test_state.board.is_on_grid(piece) :
                    nR,nC = piece
                    # If there is empty space around a piece, it has liberty
                    if grid[nR][nC] == 0 and move.point != piece:
                        return False
        return True

    def violate_ko(self, player, move):

        if not move.is_play:
            return False

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
        for r in range(board.board_height) :
            for c in range(board.board_width) :
                move = Move(point=Point(row=r,col=c))
                if self.is_valid_move(move) :
                    leg_moves.append(move)
                    
        leg_moves.append(Move.pass_turn())
        leg_moves.append(Move.resign())
        
        return leg_moves

    def is_valid_move(self, move):

        if self.is_over():
            return False

        if move.is_pass or move.is_resign:
            return True

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
        """
        if self.moves>(self.board.board_width *self.board.board_height * 2):
            return True
        """
        if self.last_move is None:
            return False
   
        if self.last_move.is_resign:
            return False
    
        if self.last_move.is_pass and self.previous_state.last_move.is_pass:
            return True
        return False
    
    def winner(self):
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
        board = self.board.grid
        visited = set()
        m = board.shape[0]
        n = board.shape[1]
        if m == 19:
            komi = 3.75
        else:
            komi = (m / 2) - 1
        #print(komi)
        count_black = -komi
        count_white = komi
        #print(count_white, count_black)
        offset = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        for i in range(m):
            for j in range(n):
                if (i, j) in visited:
                    continue
                elif board[i][j] == 1:
                    count_black += 1
                    #print("black_increase", i, j, count_black)
                elif board[i][j] == 2:
                    count_white += 1
                    #print("white_increase", i, j, count_white)
                elif board[i][j] == 0:
                    queue = set()
                    queue.add((i, j))
                    black_neighbour = False
                    white_neighbour = False
                    group_count = 0
                    while queue:
                        node_x, node_y = queue.pop()
                        if (node_x, node_y) in visited:
                            continue
                        visited.add((node_x, node_y))
                        group_count += 1
                        neighbours = offset + np.array([node_x, node_y])
                        for neighbour in neighbours:
                            if (neighbour[0], neighbour[1]) in visited:
                                continue
                            elif 0 <= neighbour[0] < m and 0 <= neighbour[1] < n:
                                val = board[neighbour[0]][neighbour[1]]
                                if val == 1:
                                    black_neighbour = True
                                elif val == 2:
                                    white_neighbour = True
                                elif val == 0:
                                    queue.add((neighbour[0], neighbour[1]))
                    if black_neighbour and white_neighbour:
                        count_black+=(group_count/2)
                        count_white+=(group_count/2)
                        pass
                    elif black_neighbour:
                        count_black += group_count
                        #print("black_group_inc", group_count, count_black)
                    elif white_neighbour:
                        count_white += group_count
                        #print("white_group_inc", group_count, count_white)
        # print(count_white, count_black)
        if count_white > count_black:
            return 2
        elif count_black > count_white:
            return 1
        else:
            return 0


class Move:
    def __init__(self, point=None, is_pass=False, is_resign=False):
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_selected = False
        self.is_resign = is_resign

    def __eq__(self, other):

        if isinstance(other,Move) :
            if self.is_play==other.is_play :
                comparison = self.point[0] == other.point[0] and self.point[1]==other.point[1] and self.is_pass==other.is_pass and self.is_resign==other.is_resign
                return comparison
            else :
                return False
        return False

    @classmethod
    def play(cls, point):
        return Move(point=point)

    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)

    @classmethod
    def ply_selected(cls):
        return Move(is_selected=True)

    @classmethod
    def resign(cls):
        return Move(is_resign=True)



"""
if __name__ == "__main__":
    BOARD_SIZE = 5


    #usage of Move class

    # Move is (1,1)
    move = Move.play(Point(row=1, col=1))
    print(move.point, move.is_play, move.is_pass, move.is_selected)  # Point(row=1, col=1) True False False

    # Move is pass
    move = Move.pass_turn()
    print(move.point, move.is_play, move.is_pass, move.is_selected)  # None False True False

    
    gamestate = GameState.new_game(BOARD_SIZE)
    print(gamestate.board.display_board())  # display the board for this gamestate

    # Set player as black
    gamestate.next_player = Player(Player.black.value)
    print(gamestate.next_player)  # Player.black

    gamestate = GameState.new_game(BOARD_SIZE)
    gamestate.board.grid = np.array(
        [[0, 1, 2, 0, 0],
         [1, 1, 2, 0, 0],
         [2, 2, 2, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    )
    print(gamestate.board.display_board())
    new_state = gamestate.apply_move(Move(Point(0, 0)))
    print(new_state.board.display_board())

 
    bot1 = RandomAgent(Player.black)
    bot2 = RandomAgent(Player.white)

    # Neural network to select the move or let's say you want to play (3,3)
    move = bot1.select_move(gamestate) or move = Move.play(Point(3,3))
    gamestate = gamestate.apply_move(move)


"""
