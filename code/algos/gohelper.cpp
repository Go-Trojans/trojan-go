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

list<std::pair<int, int>> Point::neighbours()
{
    map<int, char> action = mapAction();

    map<char, std::pair<int, int>> directions;
    list<std::pair<int, int>> neigh;
    int row, col;

    directions = mapDirections();
    size_t len = directions.size();
    for (int i = 1; i <= len; i++)
    {
        neigh.push_back(make_pair(coord.first + directions[action[i]].first, coord.second + directions[action[i]].second));
        //cout << point.first + directions[action[i]].first << " " << point.second + directions[action[i]].second << endl;
    }

    for (auto &elm : neigh)
    {
        cout << elm.first << " " << elm.second << endl;
    }

    return neigh;
}

void test()
{

    cout << "I am test" << endl;
    list<std::pair<int, int>> neigh;
    neigh = Point(1, 1).neighbours();
}

#ifdef USAGE
int main()
{

    list<std::pair<int, int>> neigh;

    neigh = Point(1, 1).neighbours();

    cout << "Print the neighbours here ..." << endl;
    for (auto &elm : neigh)
    {
        cout << elm.first << " " << elm.second << endl;
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
