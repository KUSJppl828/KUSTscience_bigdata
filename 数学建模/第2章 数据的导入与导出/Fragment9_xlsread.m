%--------------------------------------------------------------------------
%                    调用xlsread函数读取文件中的数据
%--------------------------------------------------------------------------
% CopyRight：xiezhh

num1 = xlsread('examp02_14.xls', 'A2:H4')
num2 = xlsread('examp02_14.xls', 1, 'A2:H4')
num3 = xlsread('examp02_14.xls', 'Sheet1', 'A2:H4')

convertdata = xlsread('examp02_14.xls', '', 'A2:C3', '', @setplusone1)

[num, txt, raw, X] = xlsread('examp02_14.xls', '', 'A2:H2', '', @setplusone2)