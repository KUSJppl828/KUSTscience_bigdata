#include<cstdio>

#include <cstdlib>
#include <iostream>
#include <queue>
#include <map>
#include <algorithm>
using namespace std;


int minn = 999999999, cnt = 0;
int nexts[4] = {1, 2, -1, -2};
map<string, int>mps;
struct node
{
    string str;
    int step;
    node(string st, int ss = 0)
    {
        str = st;
        step = ss;
    }
    node() {};
};
bool check(string str)
{
    return (str == "087654321" || str == "876543210");
}
int getz(string str)
{
    for(int i = 0; i < str.size(); i++)
    {
        if(str[i] == '0') return i;
    }
}
int bfs()
{
    queue<node>first;
    first.push(node("012345678", 0));
    while(!first.empty())
    {
        if(check(first.front().str))
        {
            return first.front().step;
        }
        string inits = first.front().str, ex;
        int steps = first.front().step;
        first.pop();
        int zero = getz(inits), one;
        cnt++;
        //if(first.front().step > 10 || cnt > 10000000) cout << first.front().step << "  " << cnt << endl;
        for(int i = 0; i < 4; i++)
        {
            cnt++;
            ex = inits;
            if(zero + nexts[i] > 8)
            {
                one = (zero + nexts[i]) % 9;
            }
            else if(zero + nexts[i] < 0)
            {
                one = zero + nexts[i] + 9;
            }
            else
            {
                one = zero + nexts[i];
            }
            if(zero > 8 || zero < 0 || one < 0 || one > 8)
            {
                cout << zero << " + " << nexts[i] << " = " << one << endl;
                system("pause");
            }
            swap(ex[zero], ex[one]);
            bool flag = true;
            if(mps[ex] == 0)
            {
                mps[ex] = 1;
                //cout << ex << endl;
                if(check(ex)) return steps + 1;
                first.push(node(ex, steps + 1));
            }
        }
    }
    return -1;
}
int main()
{
    printf("%d\n", bfs());
    return 0;

}