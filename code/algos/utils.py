import numpy as np
# tag::print_utils[]
from algos import gohelper

COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: ' . ',
    gohelper.Player.black: ' x ',
    gohelper.Player.white: ' o ',
}


def print_move(player, move):
    if move.is_pass:
        move_str = 'passes'
    elif move.is_resign:
        move_str = 'resigns'
    else:
        move_str = '%s%d' % (COLS[move.point.col - 1], move.point.row)
    print('%s %s' % (player, move_str))


def print_board(board):
    for row in range(board.board_width):
        bump = " " if row <= 9 else ""
        line = []
        for col in range(board.board_height):
            #stone = board.get(gohelper.Point(row=row, col=col))
            player_val = board.grid[row][col]
            print(player_val)
            if player_val == 0:
                stone = None
            elif player_val == 1:
                stone = gohelper.Player.black
            else:
                stone = gohelper.Player.white
                

            line.append(STONE_TO_CHAR[stone])
        print('%s%d %s' % (bump, row, ''.join(line)))
    print('    ' + '  '.join(COLS[:board.board_height]))
# end::print_utils[]


# tag::human_coordinates[]
def point_from_coords(coords):
    col = COLS.index(coords[0]) + 1
    row = int(coords[1:])
    return gotypes.Point(row=row, col=col)
# end::human_coordinates[]


def coords_from_point(point):
    return '%s%d' % (
        COLS[point.col - 1],
        point.row
    )

