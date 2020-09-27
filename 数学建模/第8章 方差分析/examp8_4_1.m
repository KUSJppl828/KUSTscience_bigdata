%--------------------------------------------------------------------------
%  examp8.4-1  调用manova1函数作单因素多元方差分析
%--------------------------------------------------------------------------
% CopyRight：xiezhh

% 从文件examp8_4_1.xls中读取数据
xdata = xlsread('examp8_4_1.xls');

x = [xdata(:,2:5); xdata(:,8:11)];
group = [xdata(:,6); xdata(:,12)];
% 调用manova1函数作多元方差分析
[d,p,stats] = manova1(x,group)

% 调用anova1函数对甲商品的销售额作一元方差分析
[p1,table1] = anova1(x(:,1),group)

% 调用anova1函数对乙商品的销售额作一元方差分析
[p2,table2] = anova1(x(:,2),group)

% 调用anova1函数对丙商品的销售额作一元方差分析
[p3,table3] = anova1(x(:,3),group)

% 调用anova1函数对丁商品的销售额作一元方差分析
[p4,table4] = anova1(x(:,4),group)