%--------------------------------------------------------------------------
%  第6章  参数估计与假设检验
%--------------------------------------------------------------------------
% CopyRight：xiezhh

%% examp6.1-1 常用分布参数估计
%+++++++++++++++++++++++++调用normfit函数求解+++++++++++++++++++++++++++++++
x = [15.14  14.81  15.11  15.26  15.08  15.17  15.12  14.95  15.05  14.87];
[muhat,sigmahat,muci,sigmaci] = normfit(x,0.1)

%++++++++++++++++++++++++++++调用mle函数求解++++++++++++++++++++++++++++++++
x = [15.14  14.81  15.11  15.26  15.08  15.17  15.12  14.95  15.05  14.87];
[mu_sigma,mu_sigma_ci] = mle(x,'distribution','norm','alpha',0.1)

%% examp6.1-2 自定义分布参数估计
x = [0.7917,0.8448,0.9802,0.8481,0.7627
        0.9013,0.9037,0.7399,0.7843,0.8424
        0.9842,0.7134,0.9959,0.6444,0.8362
        0.7651,0.9341,0.6515,0.7956,0.8733];
PdfFun = @(x,theta) theta*x.^(theta-1).*(x>0 & x<1);
[phat,pci] = mle(x(:),'pdf',PdfFun,'start',1)

%% examp6.1-3 多参数估计
rand('seed',1);
randn('seed',1);
x = normrnd(35,5,1000,1);
y = evrnd(20,2,1000,1);
z = randsrc(1000,1,[1,2;0.6,0.4]);
data = x.*(z==1) + y.*(z==2);
pdffun = @(t,mu1,sig1,mu2,sig2)0.6*normpdf(t,mu1,sig1)+0.4*evpdf(t,mu2,sig2);
[phat,pci] = mle(data,'pdf',pdffun,'start',[10,10,10,10],...
    'lowerbound',[-inf,0,-inf,0],'upperbound',[inf,inf,inf,inf])

%% examp6.2-1 总体标准差已知时的单个正态总体均值的检验
%++++++++++++++++++++++++++++++++双侧检验++++++++++++++++++++++++++++++++++
x = [97  102  105  112  99  103  102  94  100  95  105  98  102  100  103];
mu0 = 100;
Sigma = 2;
Alpha = 0.05;
[h,p,muci,zval] = ztest(x,mu0,Sigma,Alpha)

%++++++++++++++++++++++++++++++++单侧检验++++++++++++++++++++++++++++++++++
x = [97  102  105  112  99  103  102  94  100  95  105  98  102  100  103];
mu0 = 100;
Sigma = 2;
Alpha = 0.05;
tail = 'right';
[h,p,muci,zval] = ztest(x,mu0,Sigma,Alpha,tail)

%% examp6.2-2 总体标准差未知时的单个正态总体均值的检验
x = [49.4  50.5  50.7  51.7  49.8  47.9  49.2  51.4  48.9];
mu0 = 50;
Alpha = 0.05;
[h,p,muci,stats] = ttest(x,mu0,Alpha)

%% examp6.2-3 总体标准差未知时的两个正态总体均值的比较检验（独立样本）
x = [20.1,  20.0,  19.3,  20.6,  20.2,  19.9,  20.0,  19.9,  19.1,  19.9];
y = [18.6,  19.1,  20.0,  20.0,  20.0,  19.7,  19.9,  19.6,  20.2];
alpha = 0.05;
tail = 'both';
vartype = 'equal';
[h,p,muci,stats] = ttest2(x,y,alpha,tail,vartype)

%% examp6.2-4 总体标准差未知时的两个正态总体均值的比较检验（配对样本）
x = [80.3,68.6,72.2,71.5,72.3,70.1,74.6,73.0,58.7,78.6,85.6,78.0];
y = [74.0,71.2,66.3,65.3,66.0,61.6,68.8,72.6,65.7,72.6,77.1,71.5];
Alpha = 0.05;
tail = 'both';
[h,p,muci,stats] = ttest(x,y,Alpha,tail)

%% examp6.2-5 总体均值未知时的单个正态总体方差的检验
x = [49.4  50.5  50.7  51.7  49.8  47.9  49.2  51.4  48.9];
var0 = 1.5;
alpha = 0.05;
tail = 'both';
[h,p,varci,stats] = vartest(x,var0,alpha,tail)

%% examp6.2-6 总体均值未知时的两个正态总体方差的比较检验
x = [20.1,  20.0,  19.3,  20.6,  20.2,  19.9,  20.0,  19.9,  19.1,  19.9];
y = [18.6,  19.1,  20.0,  20.0,  20.0,  19.7,  19.9,  19.6,  20.2];
alpha = 0.05;
tail = 'both';
[h,p,varci,stats] = vartest2(x,y,alpha,tail)

%% examp6.2-7 检验功效与样本容量的计算
mu0 = 100;
sigma0 = 6.58;
mu1 = 104;
pow = 0.9;
% 调用sampsizepwr函数求样本容量n
n = sampsizepwr('z',[mu0,sigma0],mu1,pow,[],'tail','right')

n = 1:60;
% 调用sampsizepwr函数求不同的样本容量对应的检验功效
pow = sampsizepwr('z',[mu0,sigma0],mu1,[],n,'tail','right');
plot(n,pow,'k');
xlabel('样本容量');
ylabel('检验功效');

%% examp6.3-3  游程检验
x = xlsread('2012双色球开奖数据.xls',1,'I2:I98');
[h,p,stats] = runstest(x,[],'method','approximate') 

%% examp6.3-4  符号检验1
x = [-ones(69,1);zeros(23,1);ones(108,1)];
p = signtest(x)

%% examp6.3-5  符号检验2
x = [80.3,68.6,72.2,71.5,72.3,70.1,74.6,73.0,58.7,78.6,85.6,78.0];
y = [74.0,71.2,66.3,65.3,66.0,61.6,68.8,72.6,65.7,72.6,77.1,71.5];
p = signtest(x,y)

%% examp6.3-6  Wilcoxon符号秩检验
x = [20.21,19.95,20.15,20.07,19.91,19.99,20.08,20.16,...
        19.99,20.16,20.09,19.97,20.05,20.27,19.96,20.06];
[p,h,stats] = signrank(x,20)

%% examp6.3-7  Mann-Whitney秩和检验
x = [133,112,102,129,121,161,142,88,115,127,96,125];
y = [71,119,101,83,107,134,92];
[p,h,stats] = ranksum(x,y,'method','approximate')

%% examp6.3-8  分布的拟合与检验案例
%*****************************读取文件中数据********************************
score = xlsread('examp02_14.xls','Sheet1','G2:G52');
score = score(score > 0);


%*******************调用chi2gof函数进行卡方拟合优度检验***********************
[h,p,stats] = chi2gof(score)

ctrs = [50 60 70 78 85 94];
% 指定'ctrs'参数，进行卡方拟合优度检验
[h,p,stats] = chi2gof(score,'ctrs',ctrs)

% 指定'nbins'参数，进行卡方拟合优度检验
[h,p,stats] = chi2gof(score,'nbins',6)

% 指定分布为默认的正态分布，分布参数由x进行估计
[h,p,stats] = chi2gof(score,'nbins',6);

ms = mean(score);
ss = std(score);
% 指定'cdf'参数
[h,p,stats] = chi2gof(score,'nbins',6,'cdf',{'normcdf', ms, ss});
[h,p,stats] = chi2gof(score,'nbins',6,'cdf',{@normcdf, ms, ss});

% 同时指定'cdf'和'nparams'参数
[h,p,stats] = chi2gof(score,'nbins',6,'cdf',{@normcdf,ms,ss},'nparams',2)

% 调用chi2gof函数检验数据是否服从标准正态分布
[h,p] = chi2gof(score,'cdf',@normcdf)

% 指定初始分组数为6，检验总成绩数据是否服从参数为ms = 79的泊松分布
[h,p] = chi2gof(score,'nbins',6,'cdf',{@poisscdf, ms})

% 指定初始分组数为6，最小理论频数为3，检验总成绩数据是否服从正态分布
h = chi2gof(score,'nbins',6,'cdf',{@normcdf, ms, ss},'emin',3)


%*************************调用jbtest函数进行正态性检验***********************
randn('seed',0)
x = randn(10000,1);
h = jbtest(x)  % 调用jbtest函数进行正态性检验

x(end) = 5;  % 将向量x的最后一个元素改为5
h = jbtest(x)  % 再次调用jbtest函数进行正态性检验

% 调用jbtest函数对成绩数据进行Jarque-Bera检验
[h,p,jbstat,critval] = jbtest(score)


%*************************调用kstest函数进行正态性检验***********************
cdf = [score, normcdf(score, 79, 10.1489)];
% 调用kstest函数，检验总成绩是否服从由cdf指定的分布
[h,p,ksstat,cv] = kstest(score,cdf)


%*************调用kstest2函数检验两个班的总成绩是否服从相同的分布*************
banji = xlsread('examp02_14.xls','Sheet1','B2:B52');
score = xlsread('examp02_14.xls','Sheet1','G2:G52');
banji = banji(score > 0);
score = score(score > 0);
score1 = score(banji == 60101);
score2 = score(banji == 60102);
% 调用kstest2函数检验两个班的总成绩是否服从相同的分布
[h,p,ks2stat] = kstest2(score1,score2)


%*******************分别绘制两个班的总成绩的经验分布图***********************
figure;
F1 = cdfplot(score1);
set(F1,'LineWidth',2,'Color','r');
hold on;
F2 = cdfplot(score2);
set(F2,'LineStyle','-.','LineWidth',2,'Color','k');
legend('60101班总成绩的经验分布函数','60102班总成绩的经验分布函数',...
          'Location','NorthWest');


%*************************调用kstest2函数进行正态性检验***********************
randn('seed',0)
x = normrnd(mean(score),std(score),10000,1);
% 调用kstest2函数检验总成绩数据score与随机数向量x是否服从相同的分布
[h,p] = kstest2(score,x,0.05)


%**********************调用lillietest函数进行分布的检验**********************
% 调用lillietest函数进行Lilliefors检验，检验总成绩数据是否服从正态分布
[h,p,kstat,critval] = lillietest(score)

% 调用lillietest函数进行Lilliefors检验，检验总成绩数据是否服从指数分布
[h, p] = lillietest(score,0.05,'exp')


%% examp6.4-1  核密度估计案例
%*****************************读取文件中数据********************************
score = xlsread('examp02_14.xls','Sheet1','G2:G52');
score = score(score > 0);


%*****************绘制频率直方图、核密度估计图、正态分布密度图****************
% 绘制频率直方图
[f_ecdf, xc] = ecdf(score);
figure;
ecdfhist(f_ecdf, xc, 7);
hold on;
xlabel('考试成绩');
ylabel('f(x)');

% 调用ksdensity函数进行核密度估计，并绘制核密度图
[f_ks1,xi1,u1] = ksdensity(score);
plot(xi1,f_ks1,'k','linewidth',3)

% 绘制正态分布密度函数图
ms = mean(score);
ss = std(score);
f_norm = normpdf(xi1,ms,ss); 
plot(xi1,f_norm,'r-.','linewidth',3);

% 为图形加标注框
legend('频率直方图','核密度估计图', '正态分布密度图', 'Location','NorthWest');
u1    %查看默认窗宽



%**********************绘制不同窗宽对应的核密度函数图************************
[f_ks1,xi1] = ksdensity(score,'width',0.1);
[f_ks2,xi2] = ksdensity(score,'width',1);
[f_ks3,xi3] = ksdensity(score,'width',5);
[f_ks4,xi4] = ksdensity(score,'width',9);
figure;
% 分别绘制不同窗宽对应的核密度估计图，它们对应不同的线型和颜色
plot(xi1,f_ks1,'c-.','linewidth',2);
hold on;
xlabel('考试成绩');
ylabel('核密度估计');
plot(xi2,f_ks2,'r:','linewidth',2);
plot(xi3,f_ks3,'k','linewidth',2);
plot(xi4,f_ks4,'b--','linewidth',2);
legend('窗宽为0.1','窗宽为1','窗宽为5','窗宽为9','Location','NorthWest');



%**********************绘制不同核函数对应的核密度函数图**********************
[f_ks1,xi1] = ksdensity(score,'kernel','normal');
[f_ks2,xi2] = ksdensity(score,'kernel','box');
[f_ks3,xi3] = ksdensity(score,'kernel','triangle');
[f_ks4,xi4] = ksdensity(score,'kernel','epanechnikov');
figure;
% 分别绘制不同核函数对应的核密度估计图，它们对应不同的线型和颜色
plot(xi1,f_ks1,'k','linewidth',2);
hold on;
xlabel('考试成绩');
ylabel('核密度估计');
plot(xi2,f_ks2,'r:','linewidth',2);
plot(xi3,f_ks3,'b-.','linewidth',2);
plot(xi4,f_ks4,'c--','linewidth',2);
legend('Gaussian','Uniform','Triangle','Epanechnikov','Location','NorthWest');



%***************绘制经验分布函数、估计的分布函数和理论正态分布图***************
figure;
% 绘制经验分布函数图
[h,stats] = cdfplot(score);
set(h,'color','r', 'LineStyle', ':','LineWidth',2);
hold on;
title ('');
xlabel('考试成绩');
ylabel('F(x)');

% 调用ksdensity函数对累积分布函数进行估计，并绘制估计的分布函数图
[f_ks, xi] = ksdensity(score,'function','cdf');
plot(xi,f_ks,'k','linewidth',2);

% 绘制理论正态分布的分布函数曲线
y = normcdf(xi,stats.mean,stats.std);
plot(xi,y,'b-.','LineWidth',2);

legend('经验分布函数', '估计的分布函数','理论正态分布','Location','NorthWest');