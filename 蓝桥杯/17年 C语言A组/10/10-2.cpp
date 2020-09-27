#include<iostream>
using namespace std;
bool res[10001][10001]={0};
int main()
{
       int n;
       cin>>n;
       int a,b,c,d;
       for(int i=0;i<n;i++)
       {
              cin>>a>>b>>c>>d;
              if(a>c)
              {
                     int tmp=a;
                     a=c;
                     c=tmp;
              }
              if(b>d)
              {
                     int tmp=b;
                     b=d;
                     d=tmp;
              }
              for(int p=a;p<c;p++)
              {
                     for(int q=b;q<d;q++)
                            res[p][q]=1;
              }      
       }
       int num=0;
       for(int i=0;i<10001;i++)
       {
              for(int j=0;j<10001;j++)
              {
                     if(res[i][j]==1)
                        num++;
              }
       }
       cout<<num;
       return 0;
 }