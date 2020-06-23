import numpy as np
from algos import gohelper

COLUMNS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'
PLAYER_TO_CHAR = {
    None: ' . ',
    gohelper.Player.black: ' x ',
    gohelper.Player.white: ' o ',
}

def display_board(board):
    for row in range(board.board_width):
        bump = " " if row <= 9 else ""
        line = []
        for col in range(board.board_height):
            player_val = board.grid[row][col]
            if player_val == 0:
                player = None
            elif player_val == 1:
                player = gohelper.Player.black
            else:
                player = gohelper.Player.white
            line.append(PLAYER_TO_CHAR[player])
        print('%s%d %s' % (bump, row, ''.join(line)))
    print('    ' + '  '.join(COLUMNS[:board.board_height]))



def point_from_alphaNumnericMove(alphaNumnericMove):
    col = COLUMNS.index(alphaNumnericMove[0])
    row = int(alphaNumnericMove[1:])
    return gohelper.Point(row=row, col=col)


def alphaNumnericMove_from_point(point):
    return '%s%d' % (
        COLUMNS[point.col],
        point.row
    )
