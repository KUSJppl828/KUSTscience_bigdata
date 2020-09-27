%--------------------------------------------------------------------------
%  第5章  描述性统计量和统计图
%--------------------------------------------------------------------------
% CopyRight：xiezhh

%% *****************************统计量**************************************
%% 求均值
score = xlsread('examp02_14.xls','Sheet1','G2:G52');
score = score(score > 0);
score_mean = mean(score)

%% 求方差和标准差
SS1 = var(score)      % 计算(5.3-1)式的方差
SS1 = var(score,0)    % 也是计算(5.3-1)式的方差
SS2 = var(score,1)    % 计算(5.3-2)式的方差
s1 = std(score)       % 计算(5.3-3)式的标准差
s1 = std(score,0)     % 也是计算(5.3-3)式的标准差
s2 = std(score,1)     % 计算(5.3-4)式的标准差

%% 求最大值和最小值
score_max = max(score)
score_min = min(score)

%% 求极差
score_range = range(score)

%% 求中位数
score_median = median(score) 

%% 求分位数
score_m1 = quantile(score,[0.25,0.5,0.75])
score_m2 = prctile(score,[25, 50, 75])

%% 求众数
score_mode = mode(score) 

%% 求变异系数
score_cvar = std(score)/mean(score)

%% 求原点矩
A2 = mean(score.^2)

%% 求中心矩
B1 = moment(score,1)
B2 = moment(score,2)

%% 求偏度
score_skewness = skewness(score)

%% 求峰度
score_kurtosis = kurtosis(score)

%% 求协方差
XY = xlsread('examp02_14.xls','Sheet1','E2:F52');
XY = XY(all(XY>0,2),:);
covXY = cov(XY)

%% 求相关系数
Rxy = corrcoef(XY)


%% *****************************统计图**************************************
%% 箱线图
score = xlsread('examp02_14.xls','Sheet1','G2:G52');
score = score(score > 0);
figure;
boxlabel = {'考试成绩箱线图'};
boxplot(score,boxlabel,'notch','on','orientation','horizontal')
xlabel('考试成绩');

%% 频数（率）直方图
figure;
[f, xc] = ecdf(score);
ecdfhist(f, xc, 7);
xlabel('考试成绩');
ylabel('f(x)');
x = 40:0.5:100;
y = normpdf(x,mean(score),std(score));
hold on
plot(x,y,'k','LineWidth',2) 
legend('频率直方图','正态分布密度曲线','Location','NorthWest');

%% 经验分布函数图
figure;
[h,stats] = cdfplot(score)
set(h,'color','k','LineWidth',2);
x = 40:0.5:100;
y = normcdf(x,stats.mean,stats.std);
hold on
plot(x,y,':k','LineWidth',2);
legend('经验分布函数','理论正态分布','Location','NorthWest');

%% 正态概率图
figure;
normplot(score); 

%% p-p图
probplot('lognormal',score)

%% q-q图
banji = xlsread('examp02_14.xls','Sheet1','B2:B52');
score = xlsread('examp02_14.xls','Sheet1','G2:G52');
banji = banji(score > 0);
score = score(score > 0);
score1 = score(banji == 60101);
score2 = score(banji == 60102);
qqplot(score1,score2)
xlabel('60101 Quantiles');
ylabel('60102 Quantiles');

% **************************频数和频率分布表********************************
% ++++++++++++++++++++调用tabulate函数作频数和频率分布表+++++++++++++++++++++
%% examp5.5-1
x = [2  2  6  5  2  3  2  4  3  4  3  4  4  4  4  2  2
      6  0  4  7  2  5  8  3  1  3  2  5  3  6  2  3  5
      4  3  1  4  2  2  2  3  1  5  2  6  3  4  1  2  5];
tabulate(x(:))

%% examp5.5-2
x = ['If x is a numeric array, TABLE is a numeric matrix.']';
tabulate(x)

%% examp5.5-3
x = ['崔家峰';'孙乃喆';'安立群';'王洪武';'王玉杰';'高纯静';'崔家峰';
        '叶 鹏';'关泽满';'谢中华';'王宏志';'孙乃喆';'崔家峰';'谢中华'];
tabulate(x)

%% examp5.5-4
x = {'崔家峰';'孙乃喆';'安立群';'王洪武';'王玉杰';'高纯静';'崔家峰';
 '叶 鹏';'关泽满';'谢中华';'王宏志';'孙乃喆';'崔家峰';'谢中华'};
tabulate(x)

%% examp5.5-5
load fisheriris                 % 载入MATLAB自带的鸢尾花数据
species = nominal(species);     % 将字符串元胞数组species转为名义尺度数组
tabulate(species)


% ++++++++++++++++++调用自编HistRate函数作频数和频率分布表+++++++++++++++++++
%% examp5.5-1 续
x = [2  2  6  5  2  3  2  4  3  4  3  4  4  4  4  2  2
      6  0  4  7  2  5  8  3  1  3  2  5  3  6  2  3  5
      4  3  1  4  2  2  2  3  1  5  2  6  3  4  1  2  5];
HistRate(x)

%% examp5.5-2 续
x = ['If x is a numeric array, TABLE is a numeric matrix.']';
HistRate(x)

%% examp5.5-3 续
x = ['崔家峰';'孙乃喆';'安立群';'王洪武';'王玉杰';'高纯静';'崔家峰';
        '叶 鹏';'关泽满';'谢中华';'王宏志';'孙乃喆';'崔家峰';'谢中华'];
HistRate(x)

%% examp5.5-4 续
x = {'崔家峰';'孙乃喆';'安立群';'王洪武';'王玉杰';'高纯静';'崔家峰';
 '叶 鹏';'关泽满';'谢中华';'王宏志';'孙乃喆';'崔家峰';'谢中华'};
HistRate(x)

%% examp5.5-5 续
load fisheriris                 % 载入MATLAB自带的鸢尾花数据
species = nominal(species);     % 将字符串元胞数组species转为名义尺度数组
HistRate(species)