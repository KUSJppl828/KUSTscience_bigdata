%--------------------------------------------------------------------------
%  examp8.5-1  调用kruskalwallis函数作单因素非参数方差分析
%--------------------------------------------------------------------------
% CopyRight：xiezhh

A1 = [1600, 1610, 1650, 1680, 1700, 1720, 1800]';
g1 = repmat({'A1'},size(A1));

A2 = [1580, 1640, 1600, 1650, 1660]';
g2 = repmat({'A2'},size(A2));

A3 = [1460, 1550, 1600, 1620, 1640, 1610, 1540, 1620]';
g3 = repmat({'A3'},size(A3));

A4 = [1510, 1520, 1530, 1570, 1600, 1680]';
g4 = repmat({'A4'},size(A4));

life = [A1;A2;A3;A4];
group = [g1;g2;g3;g4];
% 调用kruskalwallis函数作Kruskal-Wallis检验
[p,table,stats] = kruskalwallis(life,group)

% 调用anova1函数作单因素一元方差分析
[p,table] = anova1(life,group)


A1 = [1600, 1610, 1650, 1680, 1700, 1720, 2800]';
life = [A1;A2;A3;A4];
[p,table] = kruskalwallis(life,group)

[p,table] = anova1(life,group)

[c,m,h,gnames] = multcompare(stats);
c
[gnames,num2cell(m)]
