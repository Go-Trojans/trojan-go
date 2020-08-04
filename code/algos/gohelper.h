/*
Author : Rupesh Kumar 
Date   : July 30th 2020
File   : gohelper.h

Description :  Define header files and function prototypes.
*/

#ifndef GOHELPER_H
#define GOHELPER_H

#include <iostream>
#include <map>
#include <list>

using namespace std;

#define BLACK 1
#define WHITE 2

class Player
{
public:
    int color;
    Player(int color)
    {
        this->color = color;
    }
    int opp()
    {
        return (this->color == WHITE) ? BLACK : WHITE;
    }
};

#ifdef USAGE
int main()
{
    Player player(BLACK);
    cout << player.opp() << endl;
    return 0;
}
#endif

static map<char, std::pair<int, int>> mapDirections()
{
    map<char, std::pair<int, int>> directions;
    directions[{'W'}] = make_pair(-1, 0);
    directions[{'E'}] = make_pair(1, 0);
    directions[{'S'}] = make_pair(0, -1);
    directions[{'N'}] = make_pair(0, 1);

    //cout << directions[{'W'}].first << endl;

    return directions;
}
static map<int, char> mapAction()
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
public:
    std::pair<int, int> coord;

    Point()
    {
        coord.first = -1;
        coord.second = -1;
    }

    Point(int row, int col)
    {
        coord.first = row;
        coord.second = col;
    }

    list<std::pair<int, int>> neighbours();
    bool operator==(const Point &other) const
    {
        cout << "Point == operator is called" << endl;
        return (coord.first == other.coord.first && coord.second == other.coord.second);
    }

    Point *operator=(const Point &other)
    {
        cout << "Point = operator is called" << endl;
        coord.first = other.coord.first;
        coord.second = other.coord.second;
        return this;
    }
};

/*
 Function prototypes.
 */
#ifdef USAGE
Internall it uses find_connected(board, point, player).Define both functions in gohelper.cpp bool is_point_an_eye(GoBoard *board, Point point, Player player);
#endif

#endif
