import pytest
import numpy as np

import algos.godomain as godomain
import algos.gohelper as gohelper

class TestGoHelper(object):

    def test_is_point_an_eye(self):
        test_grid = np.array(
            [[0, 1, 2, 0, 2],
            [1, 1, 2, 2, 2],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0]])
        test_board = godomain.GoBoard(5, 5, 0)
        test_board.grid = test_grid
        assert gohelper.is_point_an_eye(test_board, gohelper.Point(0, 0), gohelper.Player.black.value) == True

        assert gohelper.is_point_an_eye(test_board, gohelper.Point(3, 2), gohelper.Player.black.value) == True

        test_grid = np.array(
            [[0, 1, 2, 0, 2],
            [1, 0, 2, 2, 2],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0]])
        test_board = godomain.GoBoard(5, 5, 0)
        test_board.grid = test_grid
        assert gohelper.is_point_an_eye(test_board, gohelper.Point(3, 2), gohelper.Player.black.value) == False
        assert gohelper.is_point_an_eye(test_board, gohelper.Point(0, 0), gohelper.Player.black.value) == False

    def test_find_connected(self):
        test_grid = np.array(
            [[0, 1, 2, 0, 2],
            [1, 1, 2, 2, 2],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0]])
        test_board = godomain.GoBoard(5, 5, 0)
        test_board.grid = test_grid
        assert len(gohelper.find_connected(test_board, gohelper.Point(1, 0), gohelper.Player.black.value)) == 10