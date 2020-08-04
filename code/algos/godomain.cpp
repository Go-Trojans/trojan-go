/*
Author :  
Date   : July 30th 2020
File   : gohelper.cpp

Description : Define Go Domain Rules related helper functions and data-types.
*/

#include <iostream>

#include <list>
#include <set>
#include <vector>
#include <map>
#include <tuple>
#include <iterator>

#include <algorithm>
#include <functional>
#include <utility>

#include "godomain.h"
#include "gohelper.h"

// Move member function definitions

Move::Move()
{
    this->point.coord.first = -1;
    this->point.coord.second = -1;
    this->is_play = false;
}

Move::Move(Point point, bool is_pass, bool is_resign)
{
    this->point = point;
    this->is_play = (point.coord.first == -1 && point.coord.second == -1) ? false : true;
    this->is_pass = is_pass;
    this->is_resign = is_resign;
}

/*comparsion operator overloading */
bool Move::operator==(const Move &other) const
{
    cout << "Move == operator called" << endl;
    if (is_play == other.is_play && is_pass == other.is_pass && is_resign == other.is_resign)
    {
        if (is_play == true)
            return true;
        else
            return true;
    }
    else
        return false;
}

// GoBoard member function definitions.

/* Default Constructor */
GoBoard::GoBoard()
{
    cout << "GoBoard Parametrized Constructor called" << endl;
    board_width = M;
    board_height = N;
    moves = 0; // moves played till now.
    max_move = board_width * board_height * 2;
    /*
         dynamically allocate memory of size w*h for grid. Onus is on caller to free the memory !!!
         */
    grid = new int[board_width * board_height];
    memset(grid, 0, (board_width * board_height) * (sizeof *grid));
}

/* Parametrized Constructor */
GoBoard::GoBoard(int w, int h, int m)
{
    cout << "GoBoard Parametrized Constructor called" << endl;
    board_width = w;
    board_height = h;
    moves = m; // moves played till now.
    max_move = board_width * board_height * 2;
    /*
         dynamically allocate memory of size w*h for grid. Onus is on caller to free the memory !!!
         */
    grid = new int[w * h];
    memset(grid, 0, (w * h) * (sizeof *grid));
}

/* Deconstructor
   Deallocate GoBoard grid allocated memory 
*/
GoBoard::~GoBoard(void)
{
    cout << "Freeing memory for the grid! " << grid << endl;
    delete grid;
    grid = NULL;
}

/* 
  Copy constructor 
  usage: GoBoard board1 = board2;
     OR  GoBoard board1 = *board2;  // if board2 is a pointer.
*/
GoBoard::GoBoard(const GoBoard &board)
{
    cout << "GoBoard copy Constructor called" << endl;

    board_width = board.board_width;
    board_height = board.board_height;
    moves = board.moves;
    komi = board.komi;
    max_move = board.max_move;
    /*
    This is  needed as memory will be allocated during object creation (constructor will take care of it)
    */
    grid = new int[board_width * board_height];
    memset(grid, 0, (board_width * board_height) * (sizeof *grid));

    int *other_board_grid = board.grid;

    for (int i = 0; i < board_width; i++)
        for (int j = 0; j < board_height; j++)
            *(grid + i * N + j) = *(other_board_grid + i * N + j);
}

/*
  Assignment operator for GoBoard (deepcopy)
  usage: *new_board = *board; // both are pointers so passing the base address of GoBoard class
*/
GoBoard *GoBoard::operator=(const GoBoard &board)
{
    cout << "GoBoard Assignment opearator called" << endl;

    board_width = board.board_width;
    board_height = board.board_height;
    moves = board.moves;
    komi = board.komi;
    max_move = board.max_move;

    /*  
    DANGER !!!
    This is needed if 'this' pointer is NULL/nullptras;
         else memory would have allocated during object creation (constructor will take care of it)
    */
    // if (this)
    // {
    //     cout << "Allocating memory for grid" << endl;
    //     grid = new int[board_width * board_height]; // allocate the memory for grid for new board.
    //     memset(grid, 0, (board_width * board_height) * (sizeof *grid));
    // }

    int *other_board_grid = board.grid;

    for (int i = 0; i < board_width; i++)
        for (int j = 0; j < board_height; j++)
            *(grid + i * N + j) = *(other_board_grid + i * N + j);

    return this;
}

//TBD
void GoBoard::remove_dead_stones(Player player, Point point)
{
    cout << "remove_dead_stones" << endl;
}
//TBD
void GoBoard::place_stone(Player player, Point point)
{
    cout << "place_stone" << endl;
}

bool GoBoard::is_on_grid(Point point)
{
    if ((0 <= point.coord.first < board_width) && (0 <= point.coord.second < board_height))
        return true;
    else
        return false;
}

void GoBoard::display_board()
{
    cout << "Display the grid ..." << endl;
    for (int i = 0; i < board_width; i++)
    {
        for (int j = 0; j < board_height; j++)
        {
            cout << *(grid + i * N + j) << " ";
        }
        cout << endl;
    }
}

// GameState member functions

/* Parametrized Constructor */
GameState::GameState()
{
    this->board = NULL;
    this->next_player = -1;
    previous_state = NULL;
    this->last_move = Move();
    this->moves = 0;
}

/* Parametrized Constructor */
GameState::GameState(GoBoard *board, Player next_player, GameState *previous, Move last_move, int moves)
{
    this->board = board;             // this is the responsibility of the GoBoard user to allocate memory.
    this->next_player = next_player; // enum assignment
    previous_state = previous;
    this->last_move = last_move;
    this->moves = moves;
}

/* 
  Copy constructor 
  usage: GameState game1 = game2;
  OR     GameState game1 = *game2;
*/

GameState::GameState(const GameState &game)
{

    board = game.board; // copy constructor of GoBoard should be called
    next_player = game.next_player;
    previous_state = game.previous_state;
    last_move = game.last_move;
    moves = game.moves;
}

/* 
    Assignment  Opeator  (deepcopy)
    usage: GameState game1 = game2;
*/
GameState *GameState::operator=(const GameState &game)
{
    this->board = game.board; // Assignment operator of GoBoard should be called (this->board is alread malloc'ed)
    this->next_player = game.next_player;
    this->previous_state = game.previous_state;
    this->last_move = game.last_move;
    this->moves = game.moves;

    return this;
}

GameState *GameState::new_game(int board_size)
{
    //int w = board_size;
    //int h = board_size;
    GoBoard *board = new GoBoard(); // alloc memory for board as well as board->grid

    GameState *game = new GameState(board, Player(BLACK), NULL, Move(), 0);
    return game;
}

/* apply_move() */
GameState *GameState::apply_move(Move move)
{

    //Return the new GameState after applying the move.

    /* if we don't pass */

    cout << "In apply_move ... " << endl;
    GoBoard *next_board = new GoBoard(); // board as well as grid is malloc'ed

    if (move.is_play)
    {
        cout << "In apply_move, is_play is true ... " << endl;
        if (!(this->board->is_on_grid(move.point)) || (!is_valid_move(move)))
        {
            cout << "Invalid move " << move.point.coord.first << " " << move.point.coord.second << endl;
            return NULL;
        }

        *next_board = *(this->board); // this is a deepcopy as assignment operator should be called
        cout << "New board allocation is done " << next_board << endl;
        next_board->place_stone(this->next_player, move.point);
    }
    else
    {
        delete next_board->grid;  // free grid
        delete next_board;        // free board
        next_board = this->board; // this is a shallow copy as both board are pointing to same memory allocation
    }

    return new GameState(next_board, next_player.opp(), this, move, moves + 1);
}

//TBD
list<std::pair<int, int>> GameState::detect_neighbor_ally(GameState *Self, Player player, Point point)
{
    list<std::pair<int, int>> group_allies;

    return group_allies;
}
//TBD
list<std::pair<int, int>> GameState::ally_dfs(Player player, Point point)
{
    list<std::pair<int, int>> ally_members;

    return ally_members;
}
//TBD
bool GameState::is_suicide(Player player, Move move)
{

    return true;
}
//TBD
bool GameState::violate_ko(Player player, Move move)
{

    return false;
}
//TBD
list<std::pair<int, int>> GameState::legal_moves()
{

    list<std::pair<int, int>> leg_moves;

    return leg_moves;
}
//TBD
bool GameState::is_valid_move(Move move)
{

    return true;
}
//TBD
bool GameState::is_over()
{

    return false;
}
//TBD
int GameState::winner()
{

    return 0;
}

int main()
{
    list<std::pair<int, int>> neigh;

    neigh = Point(1, 1).neighbours();

    cout << "Print the neighbours here ..." << endl;
    for (auto &elm : neigh)
    {
        cout << elm.first << " " << elm.second << endl;
    }

    // Now Let's use Move

    Point p1(1, 5);
    Point p2 = p1;

    cout << p2.coord.first << " " << p2.coord.second << endl;

    if (p1 == p2)
    {
        cout << "Points are same" << endl;
    }
    else
    {
        cout << "Points are not same" << endl;
    }

    // Test Move
    Move move(p1, false, false);
    cout << "Move Testing ..." << endl;
    cout << move.point.coord.first << " " << move.point.coord.second << " " << move.is_play << " " << move.is_pass << endl;
    neigh = move.point.neighbours();
    cout << "Print the Move.Point.neighbours here ..." << endl;
    for (auto &elm : neigh)
    {
        cout << elm.first << " " << elm.second << endl;
    }

    // Test GoBoard
    cout << "Testing GoBoard ..." << endl;
    GoBoard *board1 = new GoBoard(5, 5, 0);
    int *grid = board1->grid;
    int i = 2, j = 2;
    *(grid + i * M + j) = 1;
    cout << board1->board_height << " " << board1 << " " << sizeof(board1) << endl;
    board1->display_board();

    // copy constructor
    GoBoard board2 = *board1; // as board1 is a pointer so need to pass address of the class object which is at *board1
    cout << "Display board2 grid (deepcopy)..." << endl;
    cout << "Board Size : " << board2.board_height << " " << board2.board_width << endl;
    board2.display_board();

    // Assignment operator for deep copy
    GoBoard *board4 = new GoBoard(5, 5, 0);
    *board4 = *board1; // deepcopy
    cout << "Display board4 grid (deepcopy)..." << endl;
    cout << "Board Size : " << board4->board_height << " " << board4->board_width << endl;
    board4->display_board();

    GoBoard *board3 = board1; // shallow copy
    cout << "Display board3 grid (shallow copy) ..." << endl;
    board3->display_board();

    cout << "Printing board1, board2, board3 grid memory locations ..." << endl;
    cout << board1->grid << " " << board2.grid << " " << board3->grid << endl;

    // delete the memory for parameterized constructor and copy constructor
    //delete board1->grid; // this will be taken care using deconstructor when object will go out off scope.
    //delete board1;
    // delete board2.grid; // this will be taken care using deconstructor when object will go out off scope.

    // Test GameState
    cout << "Testing GameState now ..." << endl;
    GameState gameD;
    GameState *game = gameD.new_game(5);
    cout << "Player is : " << game->next_player.color << endl;
    Point p(4, 4);
    game = game->apply_move(Move(p, false, false));
    cout << "New Game State after move (4,4)" << endl;
    game->board->display_board();

    // ignore now this.
    game->board = board1;
    game->board->display_board();

    cout << "New GameState called to do deepcopy of GoBoard" << endl;
    GameState *new_game = new GameState();  // allocate memory for GameState
    new_game->board = new GoBoard(5, 5, 0); // allocate memory for GoBoard
    *new_game->board = *board1;             // GoBoard Assignment operator will be called and internally it will allocate memory for grid too
    new_game->board->display_board();
    return 0;
}