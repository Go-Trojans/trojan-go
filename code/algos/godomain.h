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

    Move(Point point, bool is_pass, bool is_resign)
    {
        this->point = point;
        this->is_play = (point.coord.first == -1 && point.coord.second == -1) ? false : true;
        this->is_pass = is_pass;
        this->is_resign = is_resign;
    }

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

    bool operator==(const Move &other) const;
    /*
    bool operator==(const Move &other) const
    {
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
    */
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

    /* Constructor */
    GoBoard(int w, int h, int m)
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
    ~GoBoard(void)
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
    GoBoard(const GoBoard &board)
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

    void remove_dead_stones(Player player, Point move)
    {
        cout << "remove_dead_stones" << endl;
    }

    void place_stone(Player player, Point point)
    {
        cout << "place_stone" << endl;
    }

    bool is_on_grid(Point point)
    {
        if ((0 <= point.coord.first < board_width) && (0 <= point.coord.second < board_height))
            return true;
        else
            return false;
    }

    void display_board()
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
};

#endif
