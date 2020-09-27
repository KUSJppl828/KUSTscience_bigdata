function [n,y] = SumLeq(m)
%   [n,y] = SumLeq(m)，令 y = 1^2 + 2^2 + ... + n^2，求使得 y <= m 的最大的n
%                      和相应的y。
%   
%   Copyright xiezhh

y = 0;
i = 0;
while  y < m
    i = i + 1;
    y = y + i^2;
end
n = i-1;
y = y-i^2;