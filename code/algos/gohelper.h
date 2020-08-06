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
#include <vector>

using namespace std;

#define BLACK 1
#define WHITE 2

class Player
{
public:
    int color;
    Player()
    {
        this->color = -1;
    }
    Player(int color)
    {
        this->color = color;
    }
    int opp()
    {
        return (this->color == WHITE) ? BLACK : WHITE;
    }
};

class Point
{
public:
    std::pair<int, int> coord;

    Point();                                   // Default constructor
    Point(int row, int col);                   // Parametrized constructor
    vector<Point> neighbours();                // find this point neighbours.
    bool operator==(const Point &other) const; // comparsion operator overloading
    bool operator<(const Point &other) const;  // overload < operator
    Point *operator=(const Point &other);      // assignment operator overloading
};

static map<char, Point> mapDirections()
{
    map<char, Point> directions;
    directions[{'W'}] = Point(-1, 0);
    directions[{'E'}] = Point(1, 0);
    directions[{'S'}] = Point(0, -1);
    directions[{'N'}] = Point(0, 1);

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

/*
 Function prototypes.
 */
#ifdef USAGE
Internall it uses find_connected(board, point, player).Define both functions in gohelper.cpp bool is_point_an_eye(GoBoard *board, Point point, Player player);
#endif

#endif
