#include<bits/stdc++.h>
using namespace std;
 
int dir[4][2] = {0,1, 0,-1, 1,0, -1,0};  //四个方向 
bool vis[9][9];
int ans = 0;

void dfs(int x, int y){
	if(x == 0 || y == 0 || x == 6 || y == 6){   //找到边界，分割完成 
		ans++;
		return ;
	}
	for(int i = 0; i < 4; i++){
		int x1 = x + dir[i][0];
		int y1 = y + dir[i][1];
		int x2 = 6 - x1;
		int y2 = 6 - y1;
		if(x1 >= 0 && y1 >= 0 && x1 <= 6 && y1 <= 6){
			if(!vis[x1][y1]){
				vis[x1][y1] = vis[x2][y2] = true;
				dfs(x1,y1);
				vis[x1][y1] = vis[x2][y2] = false;  //回溯 
			}
		}
	}
}
 
int main(){
	memset(vis, false, sizeof(vis));
	vis[3][3] = 1;
	dfs(3,3);
	cout<<ans/4;
	return 0;
}