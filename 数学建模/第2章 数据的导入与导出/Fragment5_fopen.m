%--------------------------------------------------------------------------
%                         调用fopen函数打开文件
%--------------------------------------------------------------------------
% CopyRight：xiezhh

[fid, message] = fopen('xiezhh.txt')

fid = fopen('xiezhh.txt')
[filename, permission] = fopen(fid)

fid = fopen('examp02_01.txt','rt');
tline = fgets(fid, 32)
fclose(fid);
