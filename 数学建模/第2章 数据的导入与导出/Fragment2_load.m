%--------------------------------------------------------------------------
%                    调用load函数读取文件中的数据
%--------------------------------------------------------------------------
% CopyRight：xiezhh

load examp02_01.txt
load  -ascii  examp02_01.txt
x1 = load('examp02_02.txt')

x1 = load('examp02_02.txt', '-ascii');

x2 = dlmread('examp02_01.txt');

x3 = textread('examp02_01.txt');
load examp02_03.txt
load examp02_04.txt

% 用load函数载入文件examp02_05.txt中的数据，出现错误
load examp02_05.txt

% 用load函数载入文件examp02_07.txt中的数据，出现错误
load examp02_07.txt

% 用load函数载入文件examp02_10.txt中的数据，出现错误
load examp02_10.txt

% 用load函数载入文件examp02_11.txt中的数据，出现错误
load examp02_11.txt

% 用load函数载入文件examp02_12.txt中的数据
x = load('examp02_12.txt')