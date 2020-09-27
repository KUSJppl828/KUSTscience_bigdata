%--------------------------------------------------------------------------
%  examp10.3-1  读取examp10_3_1.xls中数据，进行K均值聚类
%--------------------------------------------------------------------------
% CopyRight：xiezhh

%*************************读取数据，并进行标准化变换*************************
[X, textdata] = xlsread('examp10_3_1.xls');
row = ~any(isnan(X), 2);
X = X(row, :);
countryname = textdata(3:end,1);
countryname = countryname(row);

X = zscore(X);


%*************************选取初始凝聚点，进行聚类***************************
startdata = X([8, 27, 42],:);
idx = kmeans(X,3,'Start',startdata);


%****************************** 绘制轮廓图 *********************************
[S, H] = silhouette(X,idx);

countryname(idx == 1)
countryname(idx == 2)
countryname(idx == 3)