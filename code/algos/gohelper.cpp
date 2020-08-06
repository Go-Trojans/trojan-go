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
#include <iterator>
#include <algorithm>
#include <functional>
#include <tuple>

#include "gohelper.h"

using namespace std;

// Default constructor
Point::Point()
{
    coord.first = -1;
    coord.second = -1;
}

// Parametrized constructor
Point::Point(int row, int col)
{
    coord.first = row;
    coord.second = col;
}

//object comparsion (==) operator overloading
bool Point::operator==(const Point &other) const
{
    cout << "Point == operator is called" << endl;
    return (coord.first == other.coord.first && coord.second == other.coord.second);
}

//object comparsion (<) operator overloading
bool Point::operator<(const Point &other) const
{
    cout << "Point < operator is called" << endl;
    return (coord.first < other.coord.first);
}

//Assignment(=) operator overloading
Point *Point::operator=(const Point &other)
{
    cout << "Point = operator is called" << endl;
    coord.first = other.coord.first;
    coord.second = other.coord.second;
    return this;
}

// Find the neighbours of the this point.
vector<Point> Point::neighbours()
{
    map<int, char> action = mapAction();
    map<char, Point> directions;
    vector<Point> neigh;
    int row, col;

    directions = mapDirections();
    size_t len = directions.size();
    for (int i = 1; i <= len; i++)
    {
        neigh.push_back(Point(coord.first + directions[action[i]].coord.first, coord.second + directions[action[i]].coord.second));
        //.first, coord.second + directions[action[i]].second));
        //cout << point.first + directions[action[i]].first << " " << point.second + directions[action[i]].second << endl;
    }

    for (auto &elm : neigh)
    {
        cout << elm.coord.first << " " << elm.coord.second << endl;
    }

    return neigh;
}

#ifdef USAGE
int main()
{

    vector<Point> neigh;

    neigh = Point(1, 1).neighbours();

    cout << "Print the neighbours here ..." << endl;
    for (auto &elm : neigh)
    {
        cout << elm.coord.first << " " << elm.coord.second << endl;
    }

    Point p1(1, 1);
    Point p2(1, 1);

    if (p1 == p2)
    {
        cout << "both points are equal" << endl;
    }
    else
    {
        cout << "both points are not equal" << endl;
    }

    Player player(BLACK);
    cout << "OPPOSITION : " << player.opp() << endl;

    return 0;
}
#endif
