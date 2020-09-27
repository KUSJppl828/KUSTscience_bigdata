%--------------------------------------------------------------------------
%  examp8.3-2  调用anovan函数作多因素一元方差分析
%--------------------------------------------------------------------------
% CopyRight：xiezhh

% 从文件examp8_3_2.xls中读取数据
ydata = xlsread('examp8_3_2.xls');
y = ydata(:,7);
A = ydata(:,2);
B = ydata(:,3);
C = ydata(:,4);
D = ydata(:,6);
E = ydata(:,5);
varnames = {'A','B','C','D','E'};
model = [eye(5);1 1 0 0 0;1 0 1 0 0;1 0 0 0 1]

% 调用anovan函数作多因素一元方差分析
[p,table] = anovan(y,{A,B,C,D,E},'model',model,'varnames',varnames)


%********************************重新作方差分析*****************************
model = [eye(5);1 0 1 0 0]
[p,table,stats] = anovan(y,{A,B,C,D,E},'model',model,'varnames',varnames);
p
table


%********************************多重比较**********************************
[c,m,h,gnames] = multcompare(stats,'dimension',[1 2 3 4 5]);
[mean,id] = sort(m(:,1));
gnames = gnames(id);
[{'处理','均值'};gnames(end-19:end),num2cell(mean(end-19:end))]