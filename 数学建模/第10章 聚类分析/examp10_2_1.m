%--------------------------------------------------------------------------
%  examp10.2-1  读取examp10_2_1.xls中数据，进行样品系统聚类
%--------------------------------------------------------------------------
% CopyRight：xiezhh

%***************************读取数据，并进行标准化***************************
[X,textdata] = xlsread('examp10_2_1.xls');
X = zscore(X);


%*********************调用clusterdata函数进行一步聚类************************
obslabel = textdata(2:end,1);
Taverage = clusterdata(X,'linkage','average','maxclust',3);
obslabel(Taverage == 1)
obslabel(Taverage == 2)
obslabel(Taverage == 3)


%******************************* 分步聚类 **********************************
y = pdist(X);
Z = linkage(y,'average')

obslabel = textdata(2:end,1);
H = dendrogram(Z,0,'orientation','right','labels',obslabel);
set(H,'LineWidth',2,'Color','k');
xlabel('标准化距离（类平均法）')

inconsistent0 = inconsistent(Z,40)