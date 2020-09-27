#include<bits/stdc++.h>
using namespace std;

int vis[10][10];
int s[10][10];
int sum=0;

bool judge(int x,int y) {
    if(x>=0&&x<10&&y>=0&&y<10) {
    	return false;
	}
    return true;   //表明已经不再区域内，即成功走出迷宫 
}

void dfs(int x,int y) {
    if(judge(x,y)) {  //走出迷宫 
        sum++;
        return;
    }
    if(!vis[x][y]) {
        vis[x][y]=1;
        if(s[x][y]=='U'){
        	dfs(x-1,y);
		}else if(s[x][y]=='D') {
		    dfs(x+1,y);	
		}else if(s[x][y]=='R') {
			dfs(x,y+1);
		}else if(s[x][y]=='L') {
			dfs(x,y-1);
		}
    }
}

int main() {
	int ct=0;
	string str="UDDLUULRULUURLLLRRRURRUURLDLRDRUDDDDUUUUURUDLLRRUUDURLRLDLRLULLURLLRDURDLULLRDDDUUDDUDUDLLULRDLUURRR";
	for(int i=0;i<10;i++) {   //先分成而二维数组 
		for(int j=0;j<10;j++) {
			s[i][j]=str[ct];
			ct++;
		}
	}
	for(int i=0;i<10;i++) {
		for(int j=0;j<10;j++) {
			memset(vis,0,sizeof(vis));   //注意每一次深搜后都要重新将vis数组清零 
			dfs(i,j);
		}
	}
	cout<<sum;
	return 0; 
 } 