function Vm = quad2mont2(n)
%   Vm = quad2mont(n),求球面 x^2+y^2+z^2 = 4 被圆柱面 x^2+y^2 = 2*x 所截得
%   的（含在圆柱面内的部分）立体的体积的蒙特卡洛模拟值Vm. 输入参数n是随机投点
%   的个数，可以是正整数标量或向量.
% CopyRight：xiezhh

fun = @(x,y)sqrt(4-x.^2-y.^2);    % 定义被积函数
% 求体积的蒙特卡洛模拟值
for i = 1:length(n)
    % 在矩形区域（0<=x<=2, -1<=y<=1）内随机投n(i)个点
    x = 2*rand(n(i),1);           % 点的x坐标
    y = 2*rand(n(i),1)-1;         % 点的y坐标    
    id = (x-1).^2 + y.^2 <= 1;    % 落到区域 x^2 + y^2 = 2*x 内的点的坐标索引
    Vm(i) = 8*sum(fun(x(id),y(id)))/n(i);  % 求积分的模拟值
end