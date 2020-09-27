function [V0,Vm] = quad3mont(n)
%   [V0,Vm] = quad3mont(n), 蒙特卡洛方法计算3重积分，返回理论值V0和模拟值Vm. 
%   输入参数n是随机投点的个数，可以是正整数标量或向量.
% CopyRight：xiezhh

% 计算理论积分值（传统数值算法），integral3是MATLAB R2012a才有的函数
fun = @(x,y,z)x.*y.*z;
ymin = @(x)x;
ymax = @(x)2*x;
zmin = @(x,y)x.*y;
zmax = @(x,y)2*x.*y;
V0 = integral3(fun,1,2,ymin,ymax,zmin,zmax);

% 构造被积函数，x是长为3的列向量或矩阵（行数为3），x的每一列表示3维空间中的一个点
fun = @(x)prod(x);
% 求体积的蒙特卡洛模拟值
for i = 1:length(n)
    % 在立方体（1<=x<=1, 1<=y<=4, 1<=z<=16）内随机投n(i)个点
    x = unifrnd(1,2,1,n(i));                      % x坐标
    y = unifrnd(1,4,1,n(i));                      % y坐标
    z = unifrnd(1,16,1,n(i));                     % z坐标
    X = [x;y;z];
    id = (y>=x)&(y<=2*x)&(z>=x.*y)&(z<=2*x.*y);   % 落入积分区域内点的坐标索引
    Vm(i) = (4-1)*(16-1)*sum(fun(X(:,id)))/n(i);  % 求积分的模拟值
end