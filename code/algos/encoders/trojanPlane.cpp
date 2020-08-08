#include <iostream>
#include <vector>

#include "trojanPlane.h"
using namespace std;

TrojanGoPlane::TrojanGoPlane(int board_size, int plane_size)
{
    board_width = board_size;
    board_height = board_size;
    n_planes = plane_size;
}

void TrojanGoPlane::deallocateTensor(int ***tensor)
{
    cout << "Deallocate memory used during encode tensor" << endl;
    // deallocate memory for board_tensor
    for (int i = 0; i < num_planes; i++)
    {
        for (int j = 0; j < board_width; j++)
            delete[] tensor[i][j];

        delete[] tensor[i];
    }

    delete[] tensor;
}

int ***TrojanGoPlane::encode(GameState *game_state)
{
    //int board_tensor[7][5][5] = {0};
    int ***board_tensor = new int **[num_planes];
    //int ***new_board_tensor;

    int ***new_board_tensor = new int **[num_planes];
    int Y = game_state->board->board_width;  // 7*5*(5)
    int Z = game_state->board->board_height; // 7*(5)*5

    int plane_history = 1;
    bool opp = true;
    bool myself = false;
    int iter_base_opp = 0;
    int iter_base_self = 0;
    int *board_plane = nullptr;

    // Allocate memory
    for (int i = 0; i < num_planes; i++)
    {
        board_tensor[i] = new int *[Y];
        new_board_tensor[i] = new int *[Y];
        for (int j = 0; j < Y; j++)
        {
            board_tensor[i][j] = new int[Z];
            //new_board_tensor[i][j] = new int[Z];
        }
    }
    // Initialize the 3D array pointer.
    for (int i = 0; i < Z; i++)
    {
        for (int j = 0; j < Y; j++)
        {
            memset(board_tensor[i][j], 0, sizeof(int) * Z);
            //memset(new_board_tensor[i][j], 0, sizeof(int) * Z);
        }
    }

    if (game_state->next_player.color == BLACK)
    {
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                board_tensor[curr_player][i][j] = 1;
    } // for WHITE same condition is not needed as first plane is already all with 0's

    int next_player = game_state->next_player.color;
    while (game_state && plane_history <= (n_planes - 1))
    {
        if (opp)
        {
            if (next_player == BLACK)
                board_plane = convert2to1and1to0(game_state);
            else
                board_plane = convert2to0(game_state);
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    board_tensor[base_opp_history + iter_base_opp][i][j] = board_plane[5 * i + j];
            plane_history += 1;
            iter_base_opp += 1;
            opp = false;
            myself = true;
            game_state = game_state->previous_state;
        }
        else if (myself)
        {
            if (next_player == BLACK)
                board_plane = convert2to0(game_state);
            else
                board_plane = convert2to1and1to0(game_state);
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    board_tensor[base_self_history + iter_base_self][i][j] = *(board_plane + 5 * i + j);
            plane_history += 1;
            iter_base_self += 1;
            opp = true;
            myself = false;
            game_state = game_state->previous_state;
        }
        else
        {
            cout << "Invalid encoding landing ..." << endl;
            exit(0);
        }
    }
    //return board_tensor;

    new_board_tensor[num_planes - 1] = board_tensor[0];
    int j = int((num_planes - 1) / 2);
    int k = -1;
    for (int i = 0; i < num_planes - 1; i++)
    {
        if (i % 2 == 0)
        {
            new_board_tensor[i] = board_tensor[i + (-1 * k)];
            k = k + 1;
        }
        else
        {
            new_board_tensor[i] = board_tensor[i + j];
            j = j - 1;
        }
    }

    // // deallocate memory for board_tensor
    // for (int i = 0; i < num_planes; i++)
    // {
    //     for (int j = 0; j < Y; j++)
    //         delete[] board_tensor[i][j];

    //     delete[] board_tensor[i];
    // }

    delete[] board_tensor; // only deallocating board_tensor and not other pointers which are inside it. Now new_board_tensor is referrign to same pointers.
    //s{t} = [X{t}, Y{t}, X{t−1}, Y{t−1},..., X{t−7}, Y{t−7}, C].
    return new_board_tensor; // responsibility of caller t free this memory
}

void TrojanGoPlane::printTensor(int ***A)
{
    cout << "Tensor is for Player : " << endl;
    // print the 3D array
    for (int i = 0; i < n_planes; i++)
    {
        for (int j = 0; j < board_height; j++)
        {
            for (int k = 0; k < board_width; k++)
                std::cout << A[i][j][k] << " ";

            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

#ifdef USAGE
int main()
{

    TrojanGoPlane trojanGoPlane(5, 7);
    int ***tensor;

    Point point(1, 1);
    cout << point.coord.first << " " << point.coord.second << endl;

    //GoBoard *board = new GoBoard();

    // Test GameState
    cout << "Testing GameState now ..." << endl;
    GameState gameD;
    GameState *game = gameD.new_game(5);

    cout << "Player is : " << game->next_player.color << endl;
    Point pB(4, 4);
    game = game->apply_move(Move(pB, false, false));
    cout << "New Game State after move (4,4)" << endl;
    game->board->display_board();
    tensor = trojanGoPlane.encode(game);
    trojanGoPlane.printTensor(tensor);
    trojanGoPlane.deallocateTensor(tensor);

    cout << "Player is : " << game->next_player.color << endl;
    Point pW(0, 0);
    game = game->apply_move(Move(pW, false, false));
    cout << "New Game State after move (0,0)" << endl;
    game->board->display_board();
    tensor = trojanGoPlane.encode(game); // it will allocate memory for tensor.
    trojanGoPlane.printTensor(tensor);
    trojanGoPlane.deallocateTensor(tensor); // once tensor is used; call this to deallocate memory as used by encode to allocate memory

    return 0;
}
#endif
