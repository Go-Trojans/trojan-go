import random
from godomain import Move
from algo.encoders import Point

__all__ = ['RandomBot']


class RandomBot():
    print("RandomBot selecting move ...")
    def select_move(self, game_state): 
        raise NotImplementedError() 
