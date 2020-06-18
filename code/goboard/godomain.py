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
    def __init__(self, board_width, board_height, moves=0):
        self.board_width = board_width
        self.board_height = board_width
        self.moves = moves  # <1>
        self.grid = np.zeros((board_width, board_width))
        self.komi = 0  # <2>
        self.verbose = True  # <3>
        self.max_move = board_width * board_height * 2  # <4>

    # <1> Number of moves played till now.
    # <2> placeholder of komi, will see it later on how to use
    # <3> Keeping for debugging purpose, just a knob for enabling/disabling verbose logs.
    # <4> The max moves allowed for a Go game,  Games terminate when both players pass or after 19 × 19 × 2 = 722 moves.

    def place_stone(self, player, point):  # <1>
        """
        Assuming that the valid point check has been applied prior to
        calling this function. Just sets the coordinates on grid.
        """
        r, c = point
        self.grid[r][c] = player.value
        remove_dead_stones.remove_dead_stones(self.grid, player.value)

    # <1> put the stone on the board and take care of other DS like removing dead stone, etc

    def is_on_grid(self, point):
        return 0 <= point.row < self.board_width and \
               0 <= point.col < self.board_height

    def update_total_moves(self):
        self.moves = self.moves + 1

    def display_board(self):
        print(self.grid)


# ________________________________________________________________________________________________________________

# GoState mentioned in the mail but using GameState as class name.
class GameState:
    def __init__(self, board, next_player, previous, last_move):
        self.board = board  # <1>
        self.next_player = next_player  # <2>
        self.previous_state = previous  # <3>
        self.last_move = last_move  # <4>

    # <1> board          : What is the current board
    # <2> next_player    : Who's going to make next move.
    # <3> previous_state : What was the previous GameState. Or can be referred as Parent.
    # <4> last_move      : Last move played (Move.point)

    def apply_move(self, move):
        """Return the new GameState after applying the move."""

        # if we don't pass
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board
        return GameState(next_board, self.next_player.opp, self, move)

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = GoBoard(*board_size)
        return GameState(board, Player.black, None, None)

    def is_suicide(self, player, move):
        raise NotImplementedError()

    def violate_ko(self, player, move):
        raise NotImplementedError()

    # Return all legal moves
    def legal_moves(self):
        raise NotImplementedError()

    def is_valid_move(self, move):
        raise NotImplementedError()

    def is_over(self):
        raise NotImplementedError()
    def remove_dead_pieces(self,board,piece,move = None):
        """

         :param board: A 2-Dimensional numpy array
                piece: The enemy piece for which the dead pieces have to be removed
                move: The latest move played
         :return: board: A 2-Dimensional numpy array with the dead pieces removed

         Basic intuition:
             find enemy groups neighbouring the latest move with no liberties and remove them from the board.
         """
        if not move:
            visited = set()
            m = board.shape[0]
            n = board.shape[1]
            offset = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
            for i in range(m):
                for j in range(n):
                    if board[i][j] == piece:
                        if (i, j) in visited:
                            continue
                        liberty_found = False
                        remove_group = []
                        queue = set()
                        queue.add((i, j))
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
        else:
            visited = set()
            m = board.shape[0]
            n = board.shape[1]
            offset = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
            move_neighbours = offset + move
            for move_neighbour in move_neighbours:
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

                    # print(queue,remove_group)
                    if not liberty_found:
                        while remove_group:
                            del_node_x, del_node_y = remove_group.pop()
                            board[del_node_x][del_node_y] = 0
                        # board[[remove_group]] = 0
        return board
    def winner(self,board):
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
            komi = (m / 2) - 1
        print(komi)
        count_black = -komi
        count_white = komi
        print(count_white, count_black)
        offset = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        for i in range(m):
            for j in range(n):
                if (i, j) in visited:
                    continue
                elif board[i][j] == 1:
                    count_black += 1
                    print("black_increase", i, j, count_black)
                elif board[i][j] == 2:
                    count_white += 1
                    print("white_increase", i, j, count_white)
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
                        print("black_group_inc", group_count, count_black)
                    elif white_neighbour:
                        count_white += group_count
                        print("white_group_inc", group_count, count_white)
        # print(count_white, count_black)
        if count_white > count_black:
            return 2
        elif count_black > count_white:
            return 1
        else:
            return 0


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
        return Move(is_selected=True)


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
    print(move.point, move.is_play, move.is_pass, move.is_selected)  # Point(row=1, col=1) True False False

    # Move is pass
    move = Move.pass_turn()
    print(move.point, move.is_play, move.is_pass, move.is_selected)  # None False True False

    """GameState new_game"""
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

    """
    bot1 = RandomAgent(Player.black)
    bot2 = RandomAgent(Player.white)

    # Neural network to select the move or let's say you want to play (3,3)
    move = bot1.select_move(gamestate) or move = Move.play(Point(3,3))
    gamestate = gamestate.apply_move(move)

    """

