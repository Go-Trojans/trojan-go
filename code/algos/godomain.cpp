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