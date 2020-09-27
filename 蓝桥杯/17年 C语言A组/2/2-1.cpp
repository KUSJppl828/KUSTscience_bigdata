#include<bits/stdc++.h>
using namespace std;

struct state {
	int pos,cnt;    //pos记录当前空盘位置，cnt记数，表示走了几步 
	char *s;        //当前字符串 
	state(int pos,char *s,int cnt):pos(pos),s(s),cnt(cnt){}  //构造函数 
};
struct cmp {
	bool operator()(char *a,char *b) {
		return strcmp(a,b)>0;
	}
};

int change[4]={-1,-2,1,2},cnt=1;   //change表示四个位置，左1左2，右1右2 
char *start="012345678",*end="087654321";
queue<state> q;
set<char *,cmp> check;

void swap(char *s,int a,int b) {
	char t=s[a];
	s[a]=s[b];
	s[b]=t;
}

int bfs() {
	q.push(state(0,start,0));   //初始状态入队 
	while(!q.empty()) {
		state temp=q.front();  //获取当前队头结点 
		int pos=temp.pos;
		int cnt=temp.cnt;
		char *s=temp.s;
		check.insert(s);
		if(strcmp(s,end)==0) {   //已经找到就结束 
			cout<<cnt;
			return 0;
			}
		q.pop();
		//产生新字符串入队 
		for(int i=0;i<4;i++) {
			char *t=(char*)malloc(9*sizeof(char));
			strcpy(t,s);
			int newpos=(pos+change[i]+9)%9;   //交换后新的空盘位置 
			swap(t,pos,newpos);               //交换 
			if(check.find(t)==check.end()) {   
				check.insert(t);
				q.push(state(newpos,t,cnt+1));  //新结点入队 
			}
		}
	}
}

int main() {
	bfs();
	return 0;
}