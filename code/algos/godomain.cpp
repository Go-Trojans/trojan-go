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
    // means move.point is none as in python
    this->point.coord.first = -1;
    this->point.coord.second = -1;
    this->is_play = false;
}

Move::Move(Point point)
{
    this->point = point;
    this->is_play = (point.coord.first == -1 && point.coord.second == -1) ? false : true;
    this->is_pass = false;
    this->is_resign = false;
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

/*
  ==  operator for GoBoard 
  usage: if (*new_board == *board); // both are pointers so passing the base address of GoBoard class
*/
bool GoBoard::operator==(const GoBoard &board)
{
    cout << "GoBoard == opearator called" << endl;
    bool res = true;

    if (board_width = !board.board_width || board_height != board.board_height || moves != board.moves || max_move != board.max_move)
        return false;

    int *other_board_grid = board.grid;

    for (int i = 0; i < board_width; i++)
    {
        for (int j = 0; j < board_height; j++)
        {
            if (*(grid + i * N + j) != *(other_board_grid + i * N + j))
                return false;
        }
    }

    return res;
}

void GoBoard::remove_dead_stones(Player player, Point move)
{
    //cout << "In remove_dead_stones" << endl;
    //cout << "Move point is : (" << move.coord.first << ", " << move.coord.second << ")" << endl;
    bool liberty_found = false;
    int node_x, node_y;
    int val = -1; // used for finding the liberty
    bool is_in = false;

    int *board = this->grid; // this is not board but just a grid.
    int piece = player.color;

    set<std::pair<int, int>> visited; // later will change to set<Point> here as well as other places too.

    int m = this->board_width;
    int n = this->board_height;
    piece = 3 - piece;
    // cout << "Line = " << __LINE__ << endl;
    list<std::pair<int, int>> move_neighbours = move.neighbours();

    for (auto &move_neighbour : move_neighbours)
    {
        //cout << "Line = " << __LINE__ << endl;

        int r = move_neighbour.first;
        int c = move_neighbour.second;
        Point point = Point(r, c);

        if (!is_on_grid(point))
        {
            cout << "Point is not on grid : (" << point.coord.first << ", " << point.coord.second << ")" << endl;
            continue;
        }

        //cout << "Line = " << __LINE__ << endl;
        if (*(board + m * r + c) == piece)
        {
            is_in = visited.find(make_pair(node_x, node_y)) != visited.end();
            if (is_in)
                continue;

            liberty_found = false;
            list<std::pair<int, int>> remove_group;
            set<std::pair<int, int>> queue;
            queue.insert(make_pair(r, c));

            //set<std::pair<int, int>>::iterator it = queue.begin();
            while (queue.size() > 0) // while(queue) as in python
            {
                //cout << "Line = " << __LINE__ << " Size of set is : " << queue.size() << endl;
                //same as queue.pop in python
                auto nodeIterator = queue.begin();
                node_x = nodeIterator->first;
                node_y = nodeIterator->second;
                queue.erase(nodeIterator);

                is_in = visited.find(make_pair(node_x, node_y)) != visited.end(); // check if the point is present in a set or not
                if (is_in)
                    continue;
                visited.insert(make_pair(node_x, node_y));

                remove_group.push_back(make_pair(node_x, node_y));

                list<std::pair<int, int>> neighbours = Point(node_x, node_y).neighbours();
                //cout << "Line = " << __LINE__ << endl;
                for (auto &neighbour : neighbours)
                {
                    is_in = visited.find(make_pair(neighbour.first, neighbour.second)) != visited.end();
                    if (is_in)
                        continue;
                    if (neighbour.first >= 0 && neighbour.first < m && neighbour.second >= 0 && neighbour.second < n)
                    {
                        val = *(board + m * neighbour.first + neighbour.second);
                        if (val == 0)
                            liberty_found = true;
                        if (val == piece)
                            queue.insert(make_pair(neighbour.first, neighbour.second));
                    }
                }
                //cout << "Line = " << __LINE__ << endl;
            }
            //cout << "Line = " << __LINE__ << endl;

            if (!liberty_found)
            {
                int size = remove_group.size();
                std::pair<int, int> del_node;
                while (size > 0)
                {
                    del_node = remove_group.front();
                    remove_group.pop_front();
                    *(board + m * del_node.first + del_node.second) = 0;
                    size--;
                }
            }
        }
        //cout << "Line = " << __LINE__ << endl;
    }
}

void GoBoard::place_stone(Player player, Point point)
{
    cout << "In place_stone " << endl;
    cout << "Player is : " << player.color << endl;
    int r, c;
    r = point.coord.first;
    c = point.coord.second;

    //Move move = Move().play(Point(r, c));
    //cout << "Move : (" << move.point.coord.first << ", " << move.point.coord.second << ")" << endl;
    *(grid + board_width * r + c) = player.color;
    Point move_point = Point(r, c); // move_point will be used to find the neighbours of this point.
    remove_dead_stones(player, move_point);
}

bool GoBoard::is_on_grid(Point point)
{
    if (point.coord.first >= 0 && point.coord.first < board_width && point.coord.second >= 0 && point.coord.second < board_height)
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

list<std::pair<int, int>> GameState::detect_neighbor_ally(Player player, Point point)
{
    int r, c;
    int *grid = this->board->grid;
    list<std::pair<int, int>> neighbors = point.neighbours();
    list<std::pair<int, int>> group_allies;

    for (auto &piece : neighbors)
    {
        r = piece.first;
        c = piece.second;
        if (this->board->is_on_grid(Point(r, c)))
        {
            if (*(grid + this->board->board_width * r + c) == player.color)
            {
                group_allies.push_back(make_pair(r, c));
            }
        }
    }

    return group_allies;
}

list<std::pair<int, int>> GameState::ally_dfs(Player player, Point point)
{

    std::pair<int, int> piece;
    list<std::pair<int, int>> ally_members;
    list<std::pair<int, int>> stackk;
    list<std::pair<int, int>> neighbor_allies;

    stackk.push_back(make_pair(point.coord.first, point.coord.second));
    while (stackk.size() > 0)
    {
        piece = stackk.pop_back(); // TODO : will check (error: no viable overloaded '=')
        ally_members.push_back(piece);
        neighbor_allies = this->detect_neighbor_ally(player, point);
        for (auto &ally : neighbor_allies)
        { // TODO : find element in a list (may need to overload oerator ==)
            if (!(stackk.find(make_pair(ally.first, ally.second)) != stackk.end()) &&
                !(ally_members.find(make_pair(ally.first, ally.second)) != ally_members.end()))
            {
                stackk.push_back(ally);
            }
        }
    }

    return ally_members;
}

bool GameState::is_suicide(Player player, Move move)
{
    list<std::pair<int, int>> ally_members;
    list<std::pair<int, int>> neighbors;
    int r, c;

    if (!move.is_play)
    {
        return false;
    }

    GameState *test_state = new GameState();
    test_state = this; // deepcoy and should be calling Assignemnt opeartor overloading
    int *grid = test_state->board->grid;
    Point point = move.point; // Point Assignment operator will be called

    test_state->board->place_stone(player, point);
    ally_members = test_state->ally_dfs(player, point);
    for (auto &member : ally_members)
    {
        neighbors = Point(member.first, member.second).neighbours();
        for (auto &piece : neighbors)
        {
            r = piece.first;
            c = piece.second;
            if (test_state->board->is_on_grid(Point(r, c)))
            {
                //If there is empty space around a piece, it has liberty
                if ((*(grid + this->board->board_width * r + c) == 0) && !(move.point == Point(r, c)))
                    return false;
            }
        }
    }

    return true;
}

bool GameState::violate_ko(Player player, Move move)
{
    if (!move.is_play)
        return false;

    GoBoard *test_board = new GoBoard(); // MEMLEAK: who will  free the memory !!!
    *test_board = *this->board;          // deepcopy of GoBoard()
    test_board->place_stone(player, move.point);

    GameState *prev_state = this;
    for (int i = 0; i < 8; i++)
    {
        prev_state = prev_state->previous_state;
        if (!prev_state)
            break;
        if (test_board == prev_state->board) // GoBoard == operator overloading should be called
            return true;
    }

    return false;
}

list<Move> GameState::legal_moves()
{
    list<Move> leg_moves;
    Move move;
    GoBoard *board = this->board; // just a pointer assignment.
    if (!board)
    {
        cout << "GoBoard is null which is not expected" << endl;
        return leg_moves; // not sure about this empty list
    }
    for (int r = 0; r < board->board_width; r++)
    {
        for (int c = 0; c < board->board_height; c++)
        {
            move = Move(Point(r, c));
            if (this->is_valid_move(move))
                leg_moves.push_back(move);
        }
    }
    leg_moves.push_back(Move().pass_turn());
    leg_moves.push_back(Move().resign());

    return leg_moves;
}

bool GameState::is_valid_move(Move move)
{
    if (this->is_over())
        return false;

    if (move.is_pass || move.is_resign)
        return true;

    Point point = move.point;
    int r, c;
    r = point.coord.first;
    c = point.coord.second;
    GoBoard *board = this->board;
    if (!board)
    {
        cout << "GoBoard is nullptr which is not expected" << endl;
        return false;
    }

    //check if off grid or not empty position
    if ((!(board->is_on_grid(point))) || (*(board->grid + board->board_width * r + c)) != 0)
        return false;

    //check KO or suicide
    if (this->violate_ko(this->next_player, move) || (this->is_suicide(this->next_player, move)))
        return false;

    return true;
}

// move is none if move.point is (-1, -1)
static bool is_move_none(Move move)
{
    if (move.point.coord.first == -1 && move.point.coord.second == -1)
        return true;
    return false;
}

bool GameState::is_over()
{
    if (this->moves > this->board->board_width * this->board->board_height * 2)
    {
        cout << "Game is over as max moves reached : " << this->moves << endl;
        return true;
    }

    if ((is_move_none(this->last_move)) || !(this->previous_state) || (is_move_none(this->previous_state->last_move)))
    {
        cout << "Game is not over yet as last_move is none" << endl;
        return false;
    }

    if (this->last_move.is_resign)
        return true;

    if (this->last_move.is_pass && this->previous_state->last_move.is_pass)
        return true;

    return false;
}

//TBD
int GameState::winner()
{
    // if self.last_move.is_resign :
    //     print("[DEBUG] last move was resign")
    //     return self.next_player

    // board = self.board.grid
    // visited = set()
    // m = board.shape[0]
    // n = board.shape[1]
    // if m == 19:
    //     komi = 3.75
    // else:
    //     komi = (m / 2) - 1
    // #print(komi)
    // count_black = -komi
    // count_white = komi
    // #print(count_white, count_black)
    // offset = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    // for i in range(m):
    //     for j in range(n):
    //         if (i, j) in visited:
    //             continue
    //         elif board[i][j] == 1:
    //             count_black += 1
    //             #print("black_increase", i, j, count_black)
    //         elif board[i][j] == 2:
    //             count_white += 1
    //             #print("white_increase", i, j, count_white)
    //         elif board[i][j] == 0:
    //             queue = set()
    //             queue.add((i, j))
    //             black_neighbour = False
    //             white_neighbour = False
    //             group_count = 0
    //             while queue:
    //                 node_x, node_y = queue.pop()
    //                 if (node_x, node_y) in visited:
    //                     continue
    //                 visited.add((node_x, node_y))
    //                 group_count += 1
    //                 neighbours = offset + np.array([node_x, node_y])
    //                 for neighbour in neighbours:
    //                     if (neighbour[0], neighbour[1]) in visited:
    //                         continue
    //                     elif 0 <= neighbour[0] < m and 0 <= neighbour[1] < n:
    //                         val = board[neighbour[0]][neighbour[1]]
    //                         if val == 1:
    //                             black_neighbour = True
    //                         elif val == 2:
    //                             white_neighbour = True
    //                         elif val == 0:
    //                             queue.add((neighbour[0], neighbour[1]))
    //             if black_neighbour and white_neighbour:
    //                 count_black+=(group_count/2)
    //                 count_white+=(group_count/2)
    //                 pass
    //             elif black_neighbour:
    //                 count_black += group_count
    //                 #print("black_group_inc", group_count, count_black)
    //             elif white_neighbour:
    //                 count_white += group_count
    //                 #print("white_group_inc", group_count, count_white)
    // # print(count_white, count_black)
    // if count_white > count_black:
    //     return 2
    // elif count_black > count_white:
    //     return 1
    // else:
    //     return 0

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
    Point pB(4, 4);
    game = game->apply_move(Move(pB, false, false));
    cout << "New Game State after move (4,4)" << endl;
    game->board->display_board();
    Point pW(4, 3);
    game = game->apply_move(Move(pW, false, false));
    game->board->display_board();
    Point pB2(2, 2);
    game = game->apply_move(Move(pB2, false, false));
    game->board->display_board();
    Point pW2(3, 4);
    game = game->apply_move(Move(pW2, false, false));
    cout << "This board should have removed black at 4,4 " << endl;
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