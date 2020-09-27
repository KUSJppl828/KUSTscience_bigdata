%--------------------------------------------------------------------------
%   第9章   回归分析
%--------------------------------------------------------------------------
% CopyRight：xiezhh


%% examp9.2-1 一元线性回归
ClimateData = xlsread('examp9_2_1.xls');
x = ClimateData(:, 1);
y = ClimateData(:, 5);
figure;
plot(x, y, 'k.', 'Markersize', 15);
xlabel('年平均气温(x)');
ylabel('全年日照时数(y)');
R = corrcoef(x, y)

mdl1 = LinearModel.fit(x,y)
figure;
mdl1.plot
xlabel('年平均气温(x)');
ylabel('全年日照时数(y)');
title('');
legend('原始散点','回归直线','置信区间');

xnew = [5,25]';
ynew = mdl1.predict(xnew)


Res = mdl1.Residuals;
Res_Stu = Res.Studentized;
Res_Stan = Res.Standardized;
figure;
subplot(2,3,1);
plot(Res_Stu,'kx');
refline(0,-2);
refline(0,2);
title('(a) 学生化残差图')
xlabel('观测序号');ylabel('学生化残差');
subplot(2,3,2);
mdl1.plotDiagnostics('cookd');
title('(b) Cook距离图')
xlabel('观测序号');ylabel('Cook距离');
subplot(2,3,3);
mdl1.plotDiagnostics('covratio');
title('(c) Covratio统计量图');
xlabel('观测序号');ylabel('Covratio统计量');
subplot(2,3,4);
plot(Res_Stan,'kx');
refline(0,-2);
refline(0,2);
title('(d) 标准化残差图');
xlabel('观测序号');ylabel('标准化残差');
subplot(2,3,5);
mdl1.plotDiagnostics('dffits');
title('(e) Dffits统计量图');
xlabel('观测序号');ylabel('Dffits统计量');
subplot(2,3,6);
mdl1.plotDiagnostics('leverage');
title('(f) 杠杆值图');
xlabel('观测序号');ylabel('杠杆值');

id = find(abs(Res_Stu)>2);
mdl2 = LinearModel.fit(x,y,'Exclude',id)
figure;
mdl2.plot;
xlabel('年平均气温(x)');
ylabel('全年日照时数(y)');
title('');
legend('剔除异常数据后散点','回归直线','置信区间');

figure;
subplot(2,3,1);
mdl2.plotResiduals('caseorder');
title('(a) 残差值序列图');
xlabel('观测序号');ylabel('残差');
subplot(2,3,2);
mdl2.plotResiduals('fitted');
title('(b) 残差与拟合值图');
xlabel('拟合值');ylabel('残差');
subplot(2,3,3);
plot(x,mdl2.Residuals.Raw,'kx');
line([0,25],[0,0],'color','k','linestyle',':');
title('(c) 残差与自变量图');
xlabel('自变量值');ylabel('残差');
subplot(2,3,4);
mdl2.plotResiduals('histogram');
title('(d) 残差直方图');
xlabel('残差r');ylabel('f(r)');
subplot(2,3,5);
mdl2.plotResiduals('probability');
title('(e) 残差正态概率图');
xlabel('残差');ylabel('概率');
subplot(2,3,6);
mdl2.plotResiduals('lagged');
title('(f) 残差与滞后残差图');
xlabel('滞后残差');ylabel('残差');

mdl3 = LinearModel.fit(x,y,'RobustOpts','on')
xnew = sort(x);
yhat1 = mdl1.predict(xnew);
yhat3 = mdl3.predict(xnew);
plot(x, y, 'ko');
hold on;
plot(xnew, yhat1, 'r--','linewidth',3);
plot(xnew, yhat3, 'linewidth', 3);
legend('原始数据散点','非稳健拟合回归直线','稳健拟合回归直线');
xlabel('年平均气温(x)');
ylabel('全年日照时数(y)');

%% examp9.3-1 一元非线性回归
HeadData = xlsread('examp9_3_1.xls'); 
x = HeadData(:, 4); 
y = HeadData(:, 9);
plot(x, y, 'k.');
xlabel('年龄(x)');
ylabel('头围(y)');

HeadCir2 = @(beta, x)beta(1)*exp(beta(2)./(x+beta(3)));
beta0 = [53,-0.2604,0.6276];
opt = statset;
opt.Robust = 'on';
nlm1 = NonLinearModel.fit(x,y,HeadCir2,beta0,'Options',opt)
%或nlm1 = NonLinearModel.fit(x,y,@HeadCir1,beta0,'Options',opt)

xnew = linspace(0,16,50)';
ynew = nlm1.predict(xnew);
figure;
plot(x, y, 'k.');
hold on;
plot(xnew, ynew, 'linewidth', 3);
xlabel('年龄(x)');
ylabel('头围(y)');
legend('原始数据散点','非线性回归曲线');

Alpha = 0.05;
ci1 = nlm1.coefCI(Alpha)

[yp,ypci] = nlm1.predict(xnew,'Prediction','observation');;
yup = ypci(:,2);
ydown = ypci(:,1);

figure;
hold on;
h1 = fill([xnew;flipud(xnew)],[yup;flipud(ydown)],[0.5,0.5,0.5]);
set(h1,'EdgeColor','none','FaceAlpha',0.5);

plot(xnew,yup,'r--','LineWidth',2);
plot(xnew,ydown,'b-.','LineWidth',2);
plot(xnew, yp, 'k','linewidth', 2);

grid on;
ylim([32, 57]);
xlabel('年龄(x)'); 
ylabel('头围(y)');
h2 = legend('预测区间','预测区间上限','预测区间下限','回归曲线');
set(h2, 'Location', 'SouthEast');

figure;
subplot(1,2,1);
nlm1.plotResiduals('histogram');
title('(a) 残差直方图');
xlabel('残差r');ylabel('f(r)');
subplot(1,2,2);
nlm1.plotResiduals('probability');
title('(b) 残差正态概率图');
xlabel('残差');ylabel('概率');

Res2 = nlm1.Residuals;
Res_Stu2 = Res2.Studentized;
id2 = find(abs(Res_Stu2)>2);

nlm2 = NonLinearModel.fit(x,y,HeadCir2,beta0,'Exclude',id2,'Options',opt)
xb = x;  xb(id2) = [];
yb = y;  yb(id2) = [];
ynew = nlm2.predict(xnew);
figure;
plot(xb, yb, 'k.');
hold on;
plot(xnew, ynew, 'linewidth', 3);
xlabel('年龄(x)');
ylabel('头围(y)');
legend('原始数据散点','非线性回归曲线');

%% examp9.4-1 多元线性和广义线性回归
data = xlsread('examp9_4_1.xls');
X = data(:,3:7);
y = data(:,2);
[R,P] = corrcoef([y,X])
VarNames = {'y','x1','x2','x3','x4','x5'};
matrixplot(R,'FigShap','e','FigSize','Auto', ...
    'ColorBar','on','XVar', VarNames,'YVar',VarNames);

mmdl1 = LinearModel.fit(X,y)

Rx = corrcoef(X);
VIF = diag(inv(Rx))

figure;
subplot(1,2,1);
mmdl1.plotResiduals('histogram');
title('(a) 残差直方图');
xlabel('残差r');ylabel('f(r)');
subplot(1,2,2);
mmdl1.plotResiduals('probability');
title('(b) 残差正态概率图');
xlabel('残差');ylabel('概率');

Res3 = mmdl1.Residuals;
Res_Stu3 = Res3.Studentized;
id3 = find(abs(Res_Stu3)>2);

Model = 'poly10101';
mmdl2 = LinearModel.fit(X,y,Model,'Exclude',id3)

Model = 'poly22222';
mmdl3 = LinearModel.fit(X,y,Model)

figure;
plot(y,'ko');
hold on
plot(mmdl1.predict(X),':');
plot(mmdl2.predict(X),'r-.');
plot(mmdl3.predict(X),'k');
legend('y的原始散点','5元线性回归拟合',...
    '3元线性回归拟合','完全二次回归拟合');
xlabel('y的观测序号'); 
ylabel('y'); 

mmdl4 = LinearModel.stepwise(X,y, 'poly22222')
yfitted = mmdl4.Fitted;
figure;
plot(y,'ko');
hold on
plot(yfitted,':','linewidth',2);
legend('y的原始散点','逐步回归拟合');
xlabel('y的观测序号');
ylabel('y');    

model = [0 0 0 0 0
         1 0 0 0 0
         0 1 0 0 0
         0 0 0 0 1
         2 0 0 0 0
         1 1 0 0 0
         0 1 1 0 0
         1 0 0 1 0
         0 0 0 2 0
         1 0 0 0 1
         0 1 0 0 1
         0 0 1 0 1
         0 0 0 0 2];
mmdl5 = LinearModel.fit(X,y,model)

%% examp9.5-1 多元非线性回归
modelfun = @(b,x)sqrt((x(:,1)-b(1)).^2+(x(:,2)-b(2)).^2+b(3).^2)/(60*b(4))+b(5);
% modelfun = 'y ~ sqrt((x1-b1)^2 + (x2-b2)^2 + b3^2)/(60*b4)+b5';
xyt = [500    3300    21    9
       300     200    19   29
       800    1600    14   51
      1400    2200    13   17
      1700     700    11   46
      2300    2800    14   47
      2500    1900    10   14
      2900     900    11   46
      3200    3100    17   57
      3400     100    16   49];
xy = xyt(:,1:2); Minutes = xyt(:,3); Seconds = xyt(:,4);
T = Minutes + Seconds/60;

b0 = [1000 100 1 1 1];
mnlm = NonLinearModel.fit(xy,T,modelfun,b0) 

%% examp9.6-1 多项式回归
%--------------------散点图------------------
[Data,Textdata] = xlsread('examp9_6_1.xls');
x = Data(:,1);
y = Data(:,3);
timestr = Textdata(3:end,2);
figure;
plot(x,y,'k.','Markersize',15);
set(gca,'xtick',1:2:numel(x),'xticklabel',timestr(1:2:end));
rotateticklabel(gca,'x',-30);
xlabel('时间');
ylabel('食品零售价格分类指数');

%-------------------4阶多项式拟合--------------------
[p4,S4] = polyfit(x,y,4)
r = poly2sym(p4);
r = vpa(r,5)
 
%--------------------更高阶多项式拟合---------------------
[p5,S5] = polyfit(x,y,5);
S5.normr
[p6,S6] = polyfit(x,y,6);
S6.normr
[p7,S7] = polyfit(x,y,7);
S7.normr
[p8,S8] = polyfit(x,y,8);
S8.normr
[p9,S9] = polyfit(x,y,9);
S9.normr

%-------------------拟合效果图----------------------
figure;
plot(x,y,'k.','Markersize',15);
set(gca,'xtick',1:2:numel(x),'xticklabel',timestr(1:2:end));
rotateticklabel(gca,'x',-30);
xlabel('时间');
ylabel('食品零售价格分类指数');
hold on;
yd4 = polyval(p4,x);
yd6 = polyval(p6,x);
yd8 = polyval(p8,x);
yd9 = polyval(p9,x);
plot(x,yd4,'k:+');
plot(x,yd6,'k--s');
plot(x,yd8,'k-.d');
plot(x,yd9,'k-p');
legend('原始散点','4次多项式拟合','6次多项式拟合','8次多项式拟合','9次多项式拟合')