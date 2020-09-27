%--------------------------------------------------------------------------
%  第3章   数据的预处理
%--------------------------------------------------------------------------
% CopyRight：xiezhh

%% examp3.1-1
t = linspace(0,2*pi,500)';
y = 100*sin(t);

noise = normrnd(0,15,500,1);
y = y + noise;
figure;
plot(t,y);
xlabel('t');
ylabel('y = sin(t) + 噪声');

yy1 = smooth(y,30);
figure;
plot(t,y,'k:');
hold on;
plot(t,yy1,'k','linewidth',3);
xlabel('t');
ylabel('moving');
legend('加噪波形','平滑后波形');

yy2 = smooth(y,30,'lowess');
figure;
plot(t,y,'k:');
hold on;
plot(t,yy2,'k','linewidth',3);
xlabel('t');
ylabel('lowess');
legend('加噪波形','平滑后波形');

yy3 = smooth(y,30,'rlowess');
figure;
plot(t,y,'k:');
hold on;
plot(t,yy3,'k','linewidth',3);
xlabel('t');
ylabel('rlowess');
legend('加噪波形','平滑后波形');

yy4 = smooth(y,30,'loess');
figure;
plot(t,y,'k:');
hold on;
plot(t,yy4,'k','linewidth',3);
xlabel('t');
ylabel('loess');
legend('加噪波形','平滑后波形');

yy5 = smooth(y,30,'sgolay',3);
figure;
plot(t,y,'k:');
hold on;
plot(t,yy5,'k','linewidth',3);
xlabel('t');
ylabel('sgolay');
legend('加噪波形','平滑后波形');

%% examp3.1-2
x = xlsread('examp03_02.xls');
price = x(:,4)';
figure;
plot(price,'k','LineWidth',2);

xlabel('观测序号'); ylabel('上海股市日收盘价');

output1 = smoothts(price,'b',30);
output2 = smoothts(price,'b',100);
figure;
plot(price,'.');
hold on
plot(output1,'k','LineWidth',2);
plot(output2,'k-.','LineWidth',2);
xlabel('观测序号'); ylabel('Box method');
legend('原始散点','平滑曲线(窗宽30)','平滑曲线(窗宽100)','location','northwest');

output3 = smoothts(price,'g',30);
output4 = smoothts(price,'g',100,100);
figure;
plot(price,'.');
hold on
plot(output3,'k','LineWidth',2);
plot(output4,'k-.','LineWidth',2);
xlabel('观测序号'); ylabel('Gaussian window method');
legend('原始散点','平滑曲线(窗宽30，标准差0.65)',...
          '平滑曲线(窗宽100，标准差100)','location','northwest');

output5 = smoothts(price,'e',30);
output6 = smoothts(price,'e',100);
figure;
plot(price,'.');
hold on
plot(output5,'k','LineWidth',2);
plot(output6,'k-.','LineWidth',2);
xlabel('观测序号'); ylabel('Exponential method');
legend('原始散点','平滑曲线(窗宽30)','平滑曲线(窗宽100)','location','northwest');

%% examp3.1-3
t = linspace(0,2*pi,500)';
y = 100*sin(t);
noise = normrnd(0,15,500,1);
y = y + noise;
figure;
plot(t,y);
xlabel('t');
ylabel('y = sin(t) + 噪声');

yy = medfilt1(y,30);
figure;
plot(t,y,'k:');
hold on
plot(t,yy,'k','LineWidth',3);
xlabel('t');
ylabel('中值滤波');
legend('加噪波形','平滑后波形');

%% examp3.2-1
rand('seed',1);
x = [rand(5,1), 5*rand(5,1), 10*rand(5,1), 500*rand(5,1)]

% 调用zscore函数对x进行标准化变换（按列标准化）
[xz,mu,sigma] = zscore(x)

mean(xz)
std(xz)

x0 = bsxfun(@plus, bsxfun(@times, xz, sigma), mu)    % 逆标准化变换

%% examp3.3-1
rand('seed',1);
x = [rand(5,1), 5*rand(5,1), 10*rand(5,1), 500*rand(5,1)]

% 调用rscore函数对x按列进行极差归一化变换
[R,xmin,xrange] = rscore(x)

x0 = bsxfun(@plus, bsxfun(@times, R, xrange), xmin)    % 逆极差归一化变换

%% examp3.3-2
rand('seed',1);
x = [rand(5,1), 5*rand(5,1), 10*rand(5,1), 500*rand(5,1)]

% 调用mapminmax函数对转置后的x按行进行极差归一化变换，
[y,Ps] = mapminmax(x', 0, 1);
y'    % 查看变换后矩阵

x0 = mapminmax('reverse',y,Ps)'    % 逆变换