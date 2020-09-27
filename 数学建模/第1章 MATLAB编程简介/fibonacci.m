%% examp1.6-6
function  y = fibonacci(n)
% 生成斐波那契数列的第n项
%   Copyright xiezhh

if (n < 0) | (round(n) ~= n) | ~isscalar(n)
    warning('输入参数应为非负整数标量');
    y = [];
    return;
elseif n < 2
    y = n;
else
    y = fibonacci(n-2)+fibonacci(n-1);
end