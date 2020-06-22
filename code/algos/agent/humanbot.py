import random
from algos.godomain import Move
from algos.gohelper import Point, is_point_an_eye

__all__ = ['HumanBot']


class HumanBot():
    # try push
    #print("Human selecting move ...")
    def select_move(self, game_state):
        print("HumanBot {} selecting move ...".format(game_state.next_player)) 
        """Choose a move manually."""
        human_move = input('-- ')
        human_move = human_move.strip()
        r = int(human_move[0])
        c = int(human_move[1:])
        print("Move played by human : ", r,c)
        point = Point(row=r, col=c)
        move = Move.play(point)
        return move
