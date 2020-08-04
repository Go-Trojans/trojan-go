/*
Author : Rupesh Kumar 
Date   : July 30th 2020
File   : godomainr.h

Description :  Define header files and function prototypes.
*/

#ifndef GODOMAIN_H
#define GODOMAIN_H

#include <iostream>
#include <map>
#include <list>

#include "gohelper.h"

using namespace std;

#define M 5
#define N 5

class Move
{
public:
    Point point;
    bool is_play = true;
    bool is_pass = false;
    bool is_selected = false;

    bool is_resign = false;

    Move(Point point, bool is_pass, bool is_resign); // Parametrized Construcor
    bool operator==(const Move &other) const;        // comparsion operator overloading

    Move play(Point point)
    {
        return Move(point, is_pass, is_resign);
    }

    Move pass_turn()
    {
        return Move(point, true, false);
    }

    Move resign()
    {
        return Move(point, false, true);
    }
};

class GoBoard
{
public:
    int board_width;
    int board_height;
    int moves = 0;
    float komi = 0;
    int max_move = 0;
    int *grid;

    GoBoard(int w, int h, int m);                        // Parametrized Construcor
    ~GoBoard(void);                                      // Deconstrcutor
    GoBoard(const GoBoard &board);                       // Copy Constructor
    void remove_dead_stones(Player player, Point point); // TBD
    void place_stone(Player player, Point point);        // TBD
    bool is_on_grid(Point point);
    void display_board();
};

#endif
