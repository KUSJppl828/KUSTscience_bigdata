%--------------------------------------------------------------------------
%                   调用dlmread函数读取文件中的数据
%--------------------------------------------------------------------------
% CopyRight：xiezhh

x = dlmread('examp02_03.txt')

x = dlmread('examp02_03.txt', ',', 2, 3)

x = dlmread('examp02_03.txt', ',', [1, 2, 2, 5])

x = dlmread('examp02_04.txt')

x = dlmread('examp02_05.txt')

x = dlmread('examp02_06.txt')

x = dlmread('examp02_07.txt')

x = dlmread('examp02_07.txt', ',', 2,0)

x = dlmread('examp02_08.txt', ' ', [7,0,8,8])
x = x(:, 1:4:end)

x = dlmread('examp02_09.txt')