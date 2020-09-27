%--------------------------------------------------------------------------
%                    调用textread函数读取文件中的数据
%--------------------------------------------------------------------------
% CopyRight：xiezhh

x1 = textread('examp02_01.txt');

x2 = textread('examp02_02.txt');

x3 = textread('examp02_03.txt','','delimiter',',');

[c1,c2,c3,c4,c5]=textread('examp02_04.txt','%f %f %f %f %f','delimiter',',;*');
c5

x5 = textread('examp02_05.txt','','emptyvalue',-1)

x6 = textread('examp02_06.txt','','emptyvalue',-1)

x8 = textread('examp02_08.txt','','headerlines',7)

x9 = textread('examp02_09.txt','','delimiter',', ','whitespace','+i')

x9 = textread('examp02_09.txt','','delimiter','+i,')

[c1,c2,c3,c4,c5,c6,c7,c8] = textread('examp02_09.txt',...
'%f %f %f %f %f %f %f %f','delimiter',', ','whitespace','+i');
x9 = [c1,c2,c3,c4,c5,c6,c7,c8]

[c1,c2,c3,c4,c5,c6,c7] = textread('examp02_10.txt',...
'%4d %d %2d %d %d %6.3f %s','delimiter','-,:');
[c1,c2,c3,c4,c5,c6]

format = '%s %s %s %d %s %d %s %d %s';
[c1,c2,c3,c4,c5,c6,c7,c8,c9] = textread('examp02_11.txt',format,...
'delimiter',': ');
[c4 c6 c8]