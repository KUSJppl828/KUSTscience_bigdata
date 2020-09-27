#include<bits/stdc++.h>
using namespace std;

const int maxn=1e5;
int dp[maxn], a[105];//dp[i]代表拿i个包子是否可能,1表示可能，0表示不可能 

int gcd(int a, int b) {
	return b?gcd(b,a%b):a;
}
int main() {
	int n, g, num = 0;
	cin>>n;
	for (int i = 0; i < n; i++) {
		cin >> a[i];
	}
	g = gcd(a[0], a[1]);
	for (int i = 2; i < n; i++) {
		g = gcd(g, a[i]);
	}//两两求最大公因子
	if (g != 1) cout << "INF" << endl;   //不互质，则有无限种都不能表示 
	else {
		dp[0] = 1;//0个包子肯定可以
		for (int i = 0; i < n; i++) {
			for (int j = 0; j +a[i]< maxn; j++) {
				if (dp[j]) {
					dp[j + a[i]] = 1;
				}
			}
		}
		for (int i = 0; i < maxn; i++) {
			if(!dp[i]) {
				num++;
			}
		}
		cout << num << endl;
	}
	return 0;
}