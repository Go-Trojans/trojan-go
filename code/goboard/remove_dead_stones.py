import numpy as np

def remove_dead_stones(board, piece):
    visited = set()
    m = board.shape[0]
    n = board.shape[1]
    offset = np.array([[1,0],[0,1],[-1,0],[0,-1]])
    for i in range(m):
        for j in range(n):
            if board[i][j] == piece:
                if (i,j) in visited:
                    continue
                liberty_found = False
                remove_group = []
                queue = set()
                queue.add((i,j))
                while queue:
                    node_x,node_y = queue.pop()
                    if (node_x,node_y) in visited:
                        continue
                    visited.add((node_x,node_y))
                    remove_group.append([node_x,node_y])
                    neighbours = offset + np.array([node_x,node_y])
                    for neighbour in neighbours:
                        if (neighbour[0],neighbour[1]) in visited:
                            continue
                        if 0<=neighbour[0]<m and 0<=neighbour[1]<n:
                            val = board[neighbour[0]][neighbour[1]]
                            if val == 0:
                                liberty_found = True
                            if val == piece:
                                queue.add((neighbour[0],neighbour[1]))
                if not liberty_found:
                    while remove_group:
                        del_node_x,del_node_y = remove_group.pop()
                        board[del_node_x][del_node_y] = 0
                    # board[remove_group] = 0
    return board
def remove_dead_stones_with_move(board, piece,move):
    visited = set()
    m = board.shape[0]
    n = board.shape[1]
    offset = np.array([[1,0],[0,1],[-1,0],[0,-1]])
    move_neighbours = offset + move
    for move_neighbour in move_neighbours:
        if board[move_neighbour[0]][move_neighbour[1]] == piece:
            if (move_neighbour[0],move_neighbour[1]) in visited:
                continue
            liberty_found = False
            remove_group = []
            queue = set()
            queue.add((move_neighbour[0],move_neighbour[1]))
            while queue:
                node_x,node_y = queue.pop()
                if (node_x,node_y) in visited:
                    continue
                visited.add((node_x,node_y))
                remove_group.append([node_x,node_y])
                neighbours = offset + np.array([node_x,node_y])
                for neighbour in neighbours:
                    if (neighbour[0],neighbour[1]) in visited:
                        continue
                    if 0<=neighbour[0]<m and 0<=neighbour[1]<n:
                        val = board[neighbour[0]][neighbour[1]]
                        if val == 0:
                            liberty_found = True
                        if val == piece:
                            queue.add((neighbour[0],neighbour[1]))
            # print(queue,remove_group)
            if not liberty_found:
                while remove_group:
                    del_node_x,del_node_y = remove_group.pop()
                    board[del_node_x][del_node_y] = 0
                # board[[remove_group]] = 0
    return board

if __name__ == "__main__":
    board = np.zeros((19,19))
    board[1][0] = 1
    board[0][1] = 1
    board[1][2] = 1
    board[2][1] = 1
    board[1][1] = 2
    print(board)
    print(remove_dead_stones(board,2))