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
#include <list>

#include <utility>
#include <algorithm>
#include <functional>

#include "godomain.h"
#include "gohelper.h"

map<char, std::pair<int, int>> mapDirections()
{
    map<char, std::pair<int, int>> directions;
    directions[{'W'}] = make_pair(-1, 0);
    directions[{'E'}] = make_pair(1, 0);
    directions[{'S'}] = make_pair(0, -1);
    directions[{'N'}] = make_pair(0, 1);

    //cout << directions[{'W'}].first << endl;

    return directions;
}
map<int, char> mapAction()
{
    map<int, char> action;
    action.insert(make_pair(1, 'W'));
    action.insert(make_pair(2, 'E'));
    action.insert(make_pair(3, 'S'));
    action.insert(make_pair(4, 'N'));

    return action;
}

class Point
{
    std::pair<int, int> point;

public:
    Point(int row, int col)
    {
        point.first = row;
        point.second = col;
    }

    list<std::pair<int, int>> neighbours()
    {
        map<int, char> action = mapAction();

        map<char, std::pair<int, int>> directions;
        list<std::pair<int, int>> neigh;
        int row, col;

        directions = mapDirections();
        size_t len = directions.size();
        for (int i = 1; i <= len; i++)
        {
            neigh.push_back(make_pair(point.first + directions[action[i]].first, point.second + directions[action[i]].second));
            //cout << point.first + directions[action[i]].first << " " << point.second + directions[action[i]].second << endl;
        }

        for (auto &elm : neigh)
        {
            cout << elm.first << " " << elm.second << endl;
        }

        return neigh;
    }
};

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

    Move resign()
    {
        return Move(NULL, false, true);
    }
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

    void remove_dead_stones(Player player, Point move) // check move here
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
        set<std::pair<int, int>> visited;

        // m = board.shape[0]
        int m = 5;
        // n = board.shape[1]
        int n = 5;
        // piece = 3 - piece
        int piece = 3 - piece;

        // offset = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        std::list<std::pair<int, int>> offsetV;
        offsetV.push_back(make_pair(1, 0));
        offsetV.push_back(make_pair(0, 1));
        offsetV.push_back(make_pair(-1, 0));
        offsetV.push_back(make_pair(0, -1));

        // move_neighbours = offset + move
        list<std::pair<int, int>> move_neighbours;
        for (auto &elm : offsetV)
        {
            move_neighbours.push_back(make_pair(elm.first + move.first, elm.second + move.second));
        }

        // for move_neighbour in move_neighbours:
        for (auto &move_neighbour : move_neighbours)
        {
            bool liberty_found = false;
            //     r, c = move_neighbour[0], move_neighbour[1]
            int r = move_neighbour.first;
            int c = move_neighbour.second;
            //     point = Point(row=r, col=c)
            Point point = Point(r, c);
            //     if not self.is_on_grid(point):
            //         continue
            if (!is_on_grid(point))
                continue;
            //     if board[move_neighbour[0]][move_neighbour[1]] == piece:
            if (*board(m * r + c) == piece)
            {
                //         if (move_neighbour[0], move_neighbour[1]) in visited:
                //             continue

                list < std::pair<int, int>::iterator itr = find(visited.begin(), visited.end(), make_pair(r, c));
                if (!itr.end())
                    continue;

                //         liberty_found = False
                bool liberty_found = false;
                //         remove_group = []
                list<std::pair<int, int>> remove_group;

                //         queue = set()
                set<std::pair<int, int>> queue;
                //         queue.add((move_neighbour[0], move_neighbour[1]))
                queue.insert(make_pair(r, c));
                set<std::pair<int, int>>::iterator it = queue.begin();
                //         while queue:
                for (; it != queue.end(); it++)
                {
                    //             node_x, node_y = queue.pop()
                    auto nodeIterator = queue.begin();
                    int node_x = *nodeIterator.first;
                    int node_y = *nodeIterator.second;
                    queue.erase(nodeIterator)

                        //             if (node_x, node_y) in visited:
                        //                 continue
                        list < std::pair<int, int>::iterator itr = find(visited.begin(), visited.end(), make_pair(node_x, node_y));
                    if (!itr.end())
                        continue;
                    //             visited.add((node_x, node_y))
                    visited.insert(make_pair(node_x, node_y));

                    //             remove_group.append([node_x, node_y])
                    remove_group.push_back(make_pair(node_x, node_y));
                    //             neighbours = offset + np.array([node_x, node_y])
                    list<std::pair<int, int>> neighbours;
                    for (auto &elm : offsetV)
                    {
                        neighbours.push_back(make_pair(elm.first + node_x, elm.second + node_y));
                    }
                    for (auto &neighbour : neighbours)
                    {
                        list < std::pair<int, int>::iterator itr = find(neighbours.begin(), neighbours.end(), make_pair(neighbour.first, neighbour.second));

                        if (!itr.end())
                            continue;
                        if (0 <= neighbour.first < m and 0 <= neighbour.second < n)
                        {
                            val = *board(m * neighbour.first + neighbour.second);
                            if (val == 0)
                                liberty_found = true;
                            if (val == piece)
                                queue.insert(make_pair(neighbour.first, neighbour.second));
                        }
                    }
                }

                if (!liberty_found)
                {
                    std::pair<int, int> del_node;
                    for (auto &elm : remove_group)
                    {
                        del_node = remove_group.pop_front();
                        *board(m * del_node.first + del_node.second) = 0;
                    }
                }
            }
        }
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
    GameState *(const GameState &game)
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

    // let us return a list of pairs (python using list)
    list<std::pair<int, int>> detect_neighbor_ally(GameState *Self, Player player, Point point)
    {
        list<std::pair<int, int>> group_allies;

        // grid = self.board.grid
        int *grid = self->board->grid;
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

    list<std::pair<int, int>> ally_dfs(Player player, Point point)
    {
        list<std::pair<int, int>> ally_members;

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

    list<std::pair<int, int>> legal_moves()
    {

        list<std::pair<int, int>> leg_moves;

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