function Sm = quad1mont2(n)
%   Sm = quad1mont2(n),求曲线 y = sqrt(x)与直线 y = x 所围成的阴影区域的
%   面积的蒙特卡洛模拟值Sm. 输入参数n是随机投点的个数，可以是正整数标量或向量.
% CopyRight：xiezhh

fun = @(x)sqrt(x)-x;           % 定义被积函数
% 计算阴影区域的面积的蒙特卡洛模拟值
for i = 1:length(n)
    x = rand(n(i),1);          % 随机投点
    Sm(i) = mean(fun(x));      % 积分的模拟值
end