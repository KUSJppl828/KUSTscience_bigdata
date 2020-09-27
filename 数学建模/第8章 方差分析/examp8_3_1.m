%--------------------------------------------------------------------------
%  examp8.3-1  调用anovan函数作双因素一元方差分析
%--------------------------------------------------------------------------
% CopyRight：xiezhh

yield = [38	29	36	40
         45	42	37	43
         58	46	52	51
         67	70	65	71
         62	64	61	70
         58	63	71	69];
yield = yield';
yield = yield(:);

A = strcat({'N'},num2str([ones(8,1);2*ones(8,1);3*ones(8,1)]));
B = strcat({'P'},num2str([ones(4,1);2*ones(4,1)]));
B = [B;B;B];
[A, B, num2cell(yield)]

varnames = {'A','B'};

% 调用anovan函数作双因素一元方差分析
[p,table,stats,term] = anovan(yield,{A,B},'model','full','varnames',varnames)

% 调用multcompare对各处理进行多重比较
[c,m,h,gnames] = multcompare(stats,'dimension',[1 2]);
c

[gnames, num2cell(m)]