def check_ko(row,col,board,previous_board,piece_type,board_dim) :
    #returns True if no KO rule violation


    test_board = board
    test_board[row][col] = piece_type


    remove_dead_stones(test_board,3 - piece_type)

    # violating KO Rule
    dead_stones = []
    for i in range(board_dim):
        for j in range(board_dim):
            if previous_board[i][j] == piece_type and board[i][j] != piece_type:
                dead_stones.append((i, j))
    if dead_stones and compare_board(previous_board, test_board) :
        return False
    else:
        return True
