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

int main()
{
    list<std::pair<int, int>> neigh;

    neigh = Point(1, 1).neighbours();

    cout << "Print the neighbours here ..." << endl;
    for (auto &elm : neigh)
    {
        cout << elm.first << " " << elm.second << endl;
    }

    return 0;
}