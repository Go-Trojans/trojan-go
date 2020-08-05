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
    Move();                                          // Default Construcor
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

    GoBoard();                                           // Default Construcor
    GoBoard(int w, int h, int m);                        // Parametrized Construcor
    ~GoBoard(void);                                      // Deconstrcutor
    GoBoard(const GoBoard &board);                       // Copy Constructor (usage: GoBoard *new_board = *board // if board is a pointer else just pass 'board')
    GoBoard *operator=(const GoBoard &board);            // Assignment operator overloading (usage: *new_board = *board)
    void remove_dead_stones(Player player, Point point); // TBD
    void place_stone(Player player, Point point);        // TBD
    bool is_on_grid(Point point);
    void display_board();
};

class GameState
{
public:
    GoBoard *board = NULL;
    Player next_player;
    GameState *previous_state = NULL;
    Move last_move;
    int moves = 0;

    GameState();
    GameState(GoBoard *board, Player next_player, GameState *previous,
              Move last_move, int moves);        // Parametrized Constructor
    GameState(const GameState &game);            // Copy Constructor (usage: GameState *new_game = game or GameState *new_game = *game)
    GameState *operator=(const GameState &game); // Assignment Operator (usage:   *new_game = *game )

    GameState *apply_move(Move move);
    GameState *new_game(int board_size);
    list<std::pair<int, int>> detect_neighbor_ally(Player player, Point point);
    list<std::pair<int, int>> ally_dfs(Player player, Point point);
    bool is_suicide(Player player, Move move);
    bool violate_ko(Player player, Move move);
    list<std::pair<int, int>> legal_moves();
    bool is_valid_move(Move move);
    bool is_over();
    int winner();
};

#endif
