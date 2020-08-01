/*
Author :  
Date   : July 30th 2020
File   : goboard.cpp

Description : Define Go Domain Rules.
Gamestate class is using GoBoard class.
*/

#include <iostream>
using namespace std;

// Include the STL library.
#include <vector>
#include <map>
#include <set>
#include <tuple>
#include <utility>

//import copy
//import inspect
//import numpy as np
//from algos.gohelper import Player, Point
#include "godomain.h"
#include "gohelper.h"

/*
  operator overloading: Need to re-check
  */
bool Move::operator==(const Move &other) const
{
    if (is_play == other.is_play && is_pass == other.is_pass && is_resign == other.is_resign)
    {
        if (is_play == true)
            return true;
        else
            return true;
        else return false;
    }
}

class Move
{
public:
    Point *point = NULL;
    bool is_play = true;
    bool is_pass = false;
    bool is_selected = false;

    bool is_resign = false;

    Move(Point *point, bool is_pass, bool is_resign)
    {
        point = point; // is a pointer
        is_play = point == NULL ? true : false;
        is_pass = is_pass;
        is_resign = is_resign;
    }

    Move play(Point *point)
    {
        return Move(point, is_pass, is_resign);
    }

    Move pass_turn()
    {
        return Move(NULL, true, false);
    }

    Move resign(cls) : return Move(NULL, false, true);
};

/* Compare two GoBoards are same or not in terms of grid[][] 
   usage GoBoard board1, board2;
   if (board1==board2) {}   
*/
bool GoBoard::operator==(const GoBoard &rhs) const
{
    return (grid == rhs.apple);
    // or, in C++11 (must #include <tuple>)
    // return std::tie(gird) == std::tie(rhs.grid);
}

class GoBoard
{
public:
    int board_width;
    int board_height;
    int moves = 0;
    float komi = 0;
    int max_move = 0;
    int *grid;

    /* Constructor */
    GoBoard(int w, int h, int moves)
    {
        board_width = w;
        board_height = h;
        moves = moves;
        max_move = board_width * board_height * 2;
        /*
         dynamically allocate memory of size w*h for grid. Onus is on caller to free the memory !!!
         */
        grid = new int[w * h];
        /*
         Initialize the grid with 0.
         */
        memset(grid, 0, (w * h) * (sizeof *grid));
        // for (int i = 0; i < w; i++)
        //     for (int j = 0; j < h; j++)
        //         *(grid + i * N + j) = 0;
    }

    /* Deconstructor
       Deallocate GoBoard grid allocated memory 
    */
    ~GoBoard(void)
    {
        cout << "Freeing memory for the grid!" << endl;
        delete grid;
        grid = NULL;
    }

    /* 
      Copy constructor 
      Return a copy of GoBoard (same as deepcopy in python) 
      usage: GoBoard board1 = board2;
    */
    GoBoard(const GoBoard &board)
    {
        GoBoard *new_board = new GoBoard(board.board_width, board.board_height, board.moves);
        return new_board;

        // def copy_board(self) :
        //    return copy.deepcopy(self)
    }

    void remove_dead_stones(Player player, Move move)
    {

        // """

        //     :param
        //            player: Current player
        //            move: The latest move played
        //     :return: board: A 2-Dimensional numpy array with the dead pieces removed (Not needed)

        //     Basic intuition:
        //         find enemy groups neighbouring the latest move with no liberties and remove them from the board.
        //     """
        // board = self.grid # this is not board but just a grid.
        int *board = grid;
        // piece = player.value
        int piece = player; // assuming player is either 0 or 1

        // visited = set()
        // m = board.shape[0]
        // n = board.shape[1]
        // piece = 3 - piece

        // offset = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        // move_neighbours = offset + move

        // for move_neighbour in move_neighbours:
        //     r, c = move_neighbour[0], move_neighbour[1]
        //     point = Point(row=r, col=c)
        //     if not self.is_on_grid(point):
        //         continue

        //     if board[move_neighbour[0]][move_neighbour[1]] == piece:
        //         if (move_neighbour[0], move_neighbour[1]) in visited:
        //             continue
        //         liberty_found = False
        //         remove_group = []
        //         queue = set()
        //         queue.add((move_neighbour[0], move_neighbour[1]))
        //         while queue:
        //             node_x, node_y = queue.pop()
        //             if (node_x, node_y) in visited:
        //                 continue
        //             visited.add((node_x, node_y))
        //             remove_group.append([node_x, node_y])
        //             neighbours = offset + np.array([node_x, node_y])
        //             for neighbour in neighbours:
        //                 if (neighbour[0], neighbour[1]) in visited:
        //                     continue
        //                 if 0 <= neighbour[0] < m and 0 <= neighbour[1] < n:
        //                     val = board[neighbour[0]][neighbour[1]]
        //                     if val == 0:
        //                         liberty_found = True
        //                     if val == piece:
        //                         queue.add((neighbour[0], neighbour[1]))

        //         if not liberty_found:
        //             while remove_group:
        //                 del_node_x, del_node_y = remove_group.pop()
        //                 board[del_node_x][del_node_y] = 0
    }

    void place_stone(Player player, Point point)
    {
        /* Player is a enum (0,1)
           point is a set in python. Here assunimg as pair. pair <int, int> point;
        */
        int r, c;
        r = point.first;
        c = point.second;

        move = Move.play(Point(r, c));
        grid[r][c] = player;       // assuming player as enum of 0 or 1 as of now
        move = np.array([ r, c ]); // TODO: need to see the DS of move and decide on this
        remove_dead_stones(player, move);
    }

    void is_on_grid(Point point)
    {
        if ((0 <= point.row < self.board_width and) && (0 <= point.col < self.board_height))
            return true;
        else
            return false;
    }

    void display_board()
    {
        for (int i = 0; i < board_width; i++)
            for (int j = 0; j < board_height; j++)
                cout << *(grid + i * N + j) << " ";
        cout << endl;
    }
};
/*
GoState mentioned in the mail but using GameState as class name.
*/
class GameState
{
public:
    GameBoard *board = new GameBoard;
    Player next_player;
    GameState *previous_state = NULL;
    Move last_move;
    int moves = 0;

    GameState(GameBoard *board, Player next_player, GameState *previous, Move last_move, int moves)
    {
        board = board;             // this is the responsibility of the GoBoard user to allocate memory.
        next_player = next_player; // enum assignment
        previous_state = previous;
        last_move = last_move;
        moves = moves;
    }

    /* 
      Copy constructor 
      Return a copy of GameState (same as deepcopy in python) 
      usage: GameState game1 = game2;
    */
    // TODO : it should be GameState *
    GameState(const GameState &game)
    {
        GameState *new_game = new GameState(game.board, game.next_player, game.previous_state, game.last_move, game.moves);
        return new_game;

        // def copy_board(self) :
        //    return copy.deepcopy(self)
    }

    GameState *apply_move(Move move, GameState *self)
    {
        Board *next_board = NULL;
        //Return the new GameState after applying the move.

        /* if we don't pass */
        if (move.is_play)
        {
            if ((!board.is_on_grid(move.point)) || (!is_valid_move(move)))
            {
                cout << "Invalid move " << move.point << endl;
                return NULL;
            }
            next_board = self->board; // this is a deepcopy as copy constructor should be called
            next_board->place_stone(next_player, move.point);
        }
        else
        {
            next_board = board;
        }

        return GameState(next_board, next_player.opp, self, move, moves + 1);
    }

    // TODO : I guess it should return GameState *
    GameState *new_game(int board_size)
    {
        GameState *game = new GameState;
        int w = board_size;
        int h = board_size;
        GoBoard *board = new GoBoard(w, h, 0);

        return GameState(board, Player.black, NULL, NULL, 0);
    }

    // let us return a set or vector (python using list)
    vector detect_neighbor_ally(Player player, Point point)
    {
        vector group_allies[];

        // grid = self.board.grid
        // neighbors = point.neighbors()  # Detect neighbors
        // group_allies = []

        // for piece in neighbors:
        //     if self.board.is_on_grid(piece) :
        //         nR,nC = piece
        //         if grid[nR][nC] == player.value:
        //             group_allies.append(piece)
        // return group_allies

        return group_allies;
    }

    vector ally_dfs(Player player, Point point)
    {
        vector ally_members[];

        // stack = [point]
        // ally_members = []
        // while stack:
        //     piece = stack.pop()
        //     ally_members.append(piece)
        //     neighbor_allies = self.detect_neighbor_ally(player,point)
        //     for ally in neighbor_allies:
        //         if ally not in stack and ally not in ally_members:
        //             stack.append(ally)
        // return ally_members

        return ally_members;
    }

    bool is_suicide(Player player, Move move)
    {

        // if not move.is_play:
        //     return False

        // test_state = self.copy()
        // grid = test_state.board.grid
        // point = move.point
        // test_state.board.place_stone(player,point)
        // ally_members = test_state.ally_dfs(player,point)
        // for member in ally_members:
        //     neighbors = member.neighbors()
        //     for piece in neighbors:
        //         if test_state.board.is_on_grid(piece) :
        //             nR,nC = piece
        //             if grid[nR][nC] == 0 and move.point != piece:
        //                 return False
        // return True

        return true;
    }

    bool violate_ko(Player player, Move move)
    {

        // if not move.is_play:
        //     return False

        // test_board = self.board.copy_board()
        // test_board.place_stone(player,move.point)

        // prev_state = self
        // for i in range(8) :
        //     prev_state = prev_state.previous_state
        //     if not prev_state :
        //         break
        //     if test_board == prev_state.board :
        //         return True

        // return False

        return false;
    }

    vector legal_moves()
    {

        vector leg_moves[];

        // leg_moves = []
        // board = self.board
        // for r in range(board.board_height) :
        //     for c in range(board.board_width) :
        //         move = Move(point=Point(row=r,col=c))
        //         if self.is_valid_move(move) :
        //             leg_moves.append(move)

        // leg_moves.append(Move.pass_turn())
        // leg_moves.append(Move.resign())

        // return leg_moves

        return leg_moves;
    }

    bool is_valid_move(Move move)
    {

        // if self.is_over():
        //     return False

        // if move.is_pass or move.is_resign:
        //     return True

        // point = move.point
        // r,c = point
        // board = self.board

        // #check if off grid or not empty position
        // if not board.is_on_grid(point) or board.grid[r][c] != 0 :
        //     return False
        // #check KO or suicide
        // if self.violate_ko(self.next_player,move) or self.is_suicide(self.next_player,move):
        //     return False
        // return True

        return true;
    }

    bool is_over()
    {
        // if self.moves > (self.board.board_width *self.board.board_height * 2):
        //     #print(inspect.currentframe().f_code.co_name, inspect.currentframe().f_back.f_code.co_name)
        //     print("Game is over as max moves reached : ", self.moves)
        //     return True

        // if not self.last_move or not self.previous_state or not self.previous_state.last_move:
        //     return False

        // if self.last_move.is_resign:
        //     return True

        // if self.last_move.is_pass and self.previous_state.last_move.is_pass:
        //     return True
        // return False

        return false;
    }

    int winner()
    {

        return 0;
    }
};

/*
int main()
{
    int BOARD_SIZE = 5

#usage of Move class

#Move is(1, 1)
    move = Move.play(Point(row=1, col=1))
    print(move.point, move.is_play, move.is_pass, move.is_selected)  # Point(row=1, col=1) True False False

#Move is pass
    move = Move.pass_turn()
    print(move.point, move.is_play, move.is_pass, move.is_selected)  # None False True False

    
    gamestate = GameState.new_game(BOARD_SIZE)
    print(gamestate.board.display_board())  # display the board for this gamestate

#Set player as black
    gamestate.next_player = Player(Player.black.value)
    print(gamestate.next_player)  # Player.black

    gamestate = GameState.new_game(BOARD_SIZE)
    gamestate.board.grid = np.array(
        [[0, 1, 2, 0, 0],
         [1, 1, 2, 0, 0],
         [2, 2, 2, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    )
    print(gamestate.board.display_board())
    new_state = gamestate.apply_move(Move(Point(0, 0)))
    print(new_state.board.display_board())

 
    bot1 = RandomAgent(Player.black)
    bot2 = RandomAgent(Player.white)

#Neural network to select the move or let's say you want to play (3,3)
    move = bot1.select_move(gamestate) or move = Move.play(Point(3,3))
    gamestate = gamestate.apply_move(move)


return 0;
}
*/