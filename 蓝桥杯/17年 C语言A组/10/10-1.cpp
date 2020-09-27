
#include<iostream>
#include <string.h>
#define N 10001
using namespace std;
int A[N][4];//存储矩形顶点 
int vis[N][N];//存储矩形围起来的点
void  swap(int &a,int &b)//交换函数 
{
	int t;t=a;a=b;b=a;
}
void place(int k)//矩形围起来的点之一 
{
	if(A[k][0]>A[k][2])
	{
		swap(A[k][0],A[k][2]);
		swap(A[k][1],A[k][3]);
	}
	for(int i=A[k][0];i<A[k][2];i++)
	{
		for(int j=A[k][1];j<A[k][3];j++)
		{
			if(vis[i][j]==1)//如果矩形内点已经计算，则跳过 
				continue;
			vis[i][j]=1;//置1表示矩形包括的点 
		}
	}
}
int main()
{
	int n,sum=0;
	cin>>n;
	memset(A,0,sizeof(int)*N*N);//数组全部元素置0 
	memset(vis,0,sizeof(int)*N*N);//数组全部元素置0 
	for(int i=0;i<n;i++)
	{
		cin>>A[i][0]>>A[i][1]>>A[i][2]>>A[i][3];//输入矩形顶点 
		place(i);//计算 
	}
	
	for(int i=0;i<=10000;i++)
	{
		for(int j=0;j<=10000;j++)
		{
			sum+=vis[i][j];//计算所有矩形围起来的点 
		}
	}
	cout<<sum<<endl;
	return 0;
}