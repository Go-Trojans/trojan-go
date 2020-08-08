#include <iostream>
#include <vector>

#include "../gohelper.h"
#include "../godomain.h"

using namespace std;

#define num_planes 7

#define curr_player 0
#define base_self_history 1
#define base_opp_history 1 + int((num_planes - 1) / 2)

static int *convert2to0(GameState *game_state)
{
    if (!game_state)
    {
        cout << "game_sate is NULL" << endl;
        return nullptr;
    }

    GoBoard board;
    board = *game_state->board; //deepcopy (GoBoard Assignment operator)
    //vector<Point> white_points;
    for (int i = 0; i < board.board_width; i++)
    {
        for (int j = 0; j < board.board_height; j++)
        {
            if (*(board.grid + board.board_width * i + j) == 2)
                *(board.grid + board.board_width * i + j) = 0;
        }
    }
    return board.grid;
}

static int *convert2to1and1to0(GameState *game_state)
{
    if (!game_state)
    {
        cout << "game_sate is NULL" << endl;
        return nullptr;
    }

    GoBoard board;
    board = *game_state->board; //deepcopy (GoBoard Assignment operator)
    for (int i = 0; i < board.board_width; i++)
    {
        for (int j = 0; j < board.board_height; j++)
        {
            if (*(board.grid + board.board_width * i + j) == 1)
                *(board.grid + board.board_width * i + j) = 0;
            else if (*(board.grid + board.board_width * i + j) == 2)
                *(board.grid + board.board_width * i + j) = 1;
        }
    }
    return board.grid;
}

class TrojanGoPlane
{
public:
    int board_width, board_height;
    int n_planes;

    TrojanGoPlane(int board_size, int plane_size);
    int ***encode(GameState *game_state);
    void deallocateTensor(int ***tensor);

    std::string name()
    {
        return "trojangoplane";
    }

    int
    num_points()
    {
        return board_width * board_height;
    }

    tuple<int, int, int> shape()
    {
        make_tuple(n_planes, board_height, board_width);
    }

    void printTensor(int ***A);
};
