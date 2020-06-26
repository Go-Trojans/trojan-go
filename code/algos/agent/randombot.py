import random
from algos.godomain import Move
from algos.gohelper import Point, is_point_an_eye

__all__ = ['RandomBot']


class RandomBot():
    # try push
    #print("RandomBot selecting move ...")
    def select_move(self, game_state):
        print("RandomBot {} selecting move ...".format(game_state.next_player)) 
        """Choose a random valid move that preserves our own eyes."""
        candidates = []
        rows = game_state.board.board_width
        cols = game_state.board.board_height
        points = [(i,j) for i in range(rows) for j in range(cols)]
        #print(points)
        for r, c in points:
            candidate = Point(row=r, col=c)
            if game_state.is_valid_move(Move.play(candidate)) and \
                        not is_point_an_eye(game_state.board,
                                            candidate,
                                            game_state.next_player):
                candidates.append(candidate)
        if not candidates:
            #print("returning pass")
            return Move.pass_turn()
        return Move.play(random.choice(candidates))
