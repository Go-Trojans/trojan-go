import random
from algos.godomain import Move
from algos.gohelper import Point, is_point_an_eye
from algos.utils import point_from_alphaNumnericMove

__all__ = ['HumanBot']


class HumanBot():
    # try push
    #print("Human selecting move ...")
    def select_move(self, game_state):
        print("HumanBot {} selecting move ...".format(game_state.next_player)) 
        """Choose a move manually."""
        human_move = input('-- ')
        human_move = human_move.strip()
        """
        r = int(human_move[0])
        c = int(human_move[1:])
        print("Move played by human : ", r,c)
        point = Point(row=r, col=c)
        """
        if human_move == "PASS":
            move = Move.pass_turn()
            return move

        point = point_from_alphaNumnericMove(human_move)
        move = Move.play(point)
        return move
