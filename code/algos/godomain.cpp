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

/* Constructor */
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
      Return a copy of GoBoard (same as deepcopy in python) 
      usage: GoBoard board1 = board2;
*/
GoBoard::GoBoard(const GoBoard &board)
{
    cout << "GoBoard copy Constructor called" << endl;

    board_width = board.board_width;
    board_height = board.board_height;
    moves = board.moves;
    komi = board.komi;
    max_move = board.max_move;
    grid = new int[board_width * board_height];
    memset(grid, 0, (board_width * board_height) * (sizeof *grid));

    int *other_board_grid = board.grid;

    for (int i = 0; i < board_width; i++)
        for (int j = 0; j < board_height; j++)
            *(grid + i * N + j) = *(other_board_grid + i * N + j);
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
    //board2 = board1; // deepcopy
    cout << "Display board2 grid (deepcopy)..." << endl;
    cout << "Board Size : " << board2.board_height << " " << board2.board_width << endl;
    board2.display_board();

    GoBoard *board3 = board1; // shallow copy
    cout << "Display board3 grid (shallow copy) ..." << endl;
    board3->display_board();

    cout << "Printing board1, board2, board3 grid memory locations ..." << endl;
    cout << board1->grid << " " << board2.grid << " " << board3->grid << endl;

    // delete the memory for parameterized constructor and copy constructor
    //delete board1->grid; // this will be taken care using deconstructor when object will go out off scope.
    delete board1;
    // delete board2.grid; // this will be taken care using deconstructor when object will go out off scope.

    return 0;
}