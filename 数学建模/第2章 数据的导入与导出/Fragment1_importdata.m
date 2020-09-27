%--------------------------------------------------------------------------
%                调用importdata函数读取文件中的数据
%--------------------------------------------------------------------------
% CopyRight：xiezhh

importdata('examp02_04.txt')

x = importdata('examp02_07.txt')
x.data
x.textdata

x = importdata('examp02_03.txt',';')
x{1}

x = importdata('examp02_08.txt',' ',2)

[x, s, h] = importdata('examp02_07.txt')

FileContent = importdata('examp02_10.txt')
FileContent = char(FileContent)
t = str2num(FileContent(:, 8:9))