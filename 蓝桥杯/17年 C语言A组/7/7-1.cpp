
#include<bits/stdc++.h>
using namespace std;
char s[105];
int index=0;

//((xx|xxx)x|(x|xx))xx为6

int f() {
    int maxn=0;
    int temp=0;
    while(index<strlen(s)) {
        if(s[index]=='(') {
            index++;
            temp+=f();//统计一对括号中X的最大值返回之后将它记录到当前括号中
           }
        else if(s[index]==')') {
            index++;
            break;
        }
        else if(s[index]=='|') {
            index++;
            if(temp>maxn) {
            	maxn=temp; //做出抉择
			} 
            temp=0; // 重新计数
        }
        else {
            temp++;
            index++;
        }
    }
    if(temp>maxn) maxn=temp;
    return maxn;
}
int main() {  
    scanf("%s",s);
    cout<<f()<<endl;
    return 0;
}