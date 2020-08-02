#include <iostream>
#include <list>
#include <set>
#include <vector>
#include <map>
#include <iterator>
#include <algorithm>
#include <functional>
#include <tuple>

using namespace std;

void printlist()
{
    std::list<std::pair<int, int>> list1;

    list1.push_back(make_pair(10, 11));
    list1.push_back(make_pair(20, 21));

    for (auto &elm : list1)
    {
        cout << elm.first << " " << elm.second;
    }
    cout << endl;
}

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

    bool operator==(const Point &other) const
    {
        return (point.first == other.point.first && point.second == other.point.second);

        //return self.row == other.row and self.col == other.col return False
    }
};

/*
  operator overloading: Need to re-check
  */
// bool Move::operator==(const Move &other) const
// {
//     if (is_play == other.is_play && is_pass == other.is_pass && is_resign == other.is_resign)
//     {
//         if (is_play == true)
//             return true;
//         else
//             return true;
//         else return false;
//     }
// }

// class Move
// {
// public:
//     Point *point = NULL;
//     bool is_play = true;
//     bool is_pass = false;
//     bool is_selected = false;

//     bool is_resign = false;

//     Move(Point *point, bool is_pass, bool is_resign)
//     {
//         point = point; // is a pointer
//         is_play = point == NULL ? true : false;
//         is_pass = is_pass;
//         is_resign = is_resign;
//     }

//     Move play(Point *point)
//     {
//         return Move(point, is_pass, is_resign);
//     }

//     Move pass_turn()
//     {
//         return Move(NULL, true, false);
//     }

//     Move resign()
//     {

//         return Move(NULL, false, true);
//     }
// };

int main()
{

    printlist();
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

    return 0;
}