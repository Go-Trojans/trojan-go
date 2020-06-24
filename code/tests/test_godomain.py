import pytest
import numpy as np
import unittest
import algos.godomain as godomain
import algos.gohelper as gohelper

class TestGoDomain(object):

    def test_violate_ko(self) :

        boards =  [godomain.GoBoard(5,5,0) for i in range(5)]
        boards[0].grid = np.zeros((5,5))
        boards[1].grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        boards[2].grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 2, 0, 0],
            [1, 2, 0, 2, 0],
            [0, 1, 2, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        boards[3].grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 2, 0, 0],
            [1, 0, 1, 2, 0],
            [0, 1, 2, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        boards[4].grid = np.zeros((5,5))


        states = [godomain.GameState(boards[0],1,None,None)]
        player = godomain.Player.black
        for i in range(1,5) :
            state = godomain.GameState(boards[i],player.opp,states[i-1],None)
            states.append(state)
        move = godomain.Move(gohelper.Point(row=2,col=1))
        assert states[2].violate_ko(godomain.Player.black, godomain.Move(gohelper.Point(row=4, col=1))) == False
        assert states[4].violate_ko(godomain.Player.black, godomain.Move(gohelper.Point(row=1, col=1))) == True
        assert states[3].violate_ko(godomain.Player.white,move) == True

    def test_is_suicide(self) :

        boards =  [godomain.GoBoard(5,5,0) for i in range(6)]
        boards[0].grid = np.zeros((5,5))
        boards[1].grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        boards[2].grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        boards[3].grid = np.array([
            [0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        boards[4].grid = np.array([
            [0, 2, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        boards[5].grid = np.array([
            [0, 2, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        states = [godomain.GameState(boards[0],1,None,None)]
        player = godomain.Player.black
        for i in range(1,5) :
            state = godomain.GameState(boards[i],player.opp,states[i-1],None)
            states.append(state)
        move = godomain.Move(gohelper.Point(row=2,col=1))
        assert states[1].is_suicide(godomain.Player.white, godomain.Move(gohelper.Point(row=2, col=1))) == False
        assert states[2].is_suicide(godomain.Player.white, godomain.Move(gohelper.Point(row=2, col=1))) == True
        assert states[3].is_suicide(godomain.Player.white, godomain.Move(gohelper.Point(row=0, col=0))) == True
        assert states[4].is_suicide(godomain.Player.black, godomain.Move(gohelper.Point(row=0, col=0))) == True
        assert states[4].is_suicide(godomain.Player.white, godomain.Move(gohelper.Point(row=0, col=0))) == False

    def test_is_valid_move(self):

        boards = [godomain.GoBoard(5, 5, 0) for i in range(5)]
        boards[0].grid = np.zeros((5, 5))
        boards[1].grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        boards[2].grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        boards[3].grid = np.array([
            [0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        boards[4].grid = np.array([
            [0, 2, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        states = [godomain.GameState(boards[0], 2, None, None)]
        player = godomain.Player.white
        for i in range(1, 5):
            state = godomain.GameState(boards[i], player.opp, states[i - 1], None)
            states.append(state)
            player = player.opp
        move = godomain.Move(gohelper.Point(row=2, col=1))
        assert states[1].is_valid_move(godomain.Move(gohelper.Point(row=1, col=1))) == False
        assert states[2].is_valid_move(godomain.Move(gohelper.Point(row=2, col=1))) == False
        assert states[3].is_valid_move(godomain.Move(gohelper.Point(row=0, col=0))) == True
        assert states[4].is_valid_move(godomain.Move(gohelper.Point(row=0, col=0))) == True
        assert states[4].is_valid_move(godomain.Move(gohelper.Point(row=0, col=1))) == False

    def test_remove_dead_stones(self):
        board = godomain.GoBoard(5, 5, 0)
        board.grid= np.array([
            [1, 0, 2, 0, 0],
            [2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
            ])
        gs = godomain.GameState(board, godomain.Player.black, None, None)
        move = godomain.Move(gohelper.Point(row=0, col=1))
        assert gs.is_suicide(godomain.Player.black, move) == True
        with pytest.raises(ValueError):
            gs = gs.apply_move(move)
            gs.board.display_board()

if __name__ == "__main__":
    unittest.main()