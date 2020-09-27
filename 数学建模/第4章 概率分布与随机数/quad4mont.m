function [V0,Vm] = quad4mont(n)
%   [V0,Vm] = quad4mont(n), 蒙特卡洛方法计算4重积分，返回理论值V0和模拟值Vm. 
%   输入参数n是随机投点的个数，可以是正整数标量或向量.
% CopyRight：xiezhh

% 计算理论积分值（传统数值算法），integral和integral3是MATLAB R2012a才有的函数
fun = @(x1,x2,x3,x4)exp(x1.*x2.*x3.*x4);
fun = @(x)arrayfun(@(x1)integral3(@(x2,x3,x4)fun(x1,x2,x3,x4),0,1,0,1,0,1),x);
V0 = integral(fun,0,1);

fun = @(x)exp(prod(x,2));  % 定义被积函数
% 求体积的蒙特卡洛模拟值
for i = 1:length(n)
    x = rand(n(i),4);      % 随机生成n(i)个4维单位超立方体内的点
    Vm(i) = mean(fun(x));  % 求积分的模拟值
end