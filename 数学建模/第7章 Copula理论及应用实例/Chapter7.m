%--------------------------------------------------------------------------
%   第7章   Copula理论及应用实例
%--------------------------------------------------------------------------
% CopyRight：xiezhh

%% examp7.4-1
%******************************读取数据*************************************
hushi = xlsread('hushi.xls');
X = hushi(:,5);
shenshi = xlsread('shenshi.xls');
Y = shenshi(:,5);


%****************************绘制频率直方图*********************************
[fx, xc] = ecdf(X);
figure;
ecdfhist(fx, xc, 30);
xlabel('沪市日收益率');
ylabel('f(x)');
[fy, yc] = ecdf(Y);
figure;
ecdfhist(fy, yc, 30);
xlabel('深市日收益率');
ylabel('f(y)');


%****************************计算偏度和峰度*********************************
xs = skewness(X)
ys = skewness(Y)

kx = kurtosis(X)
ky = kurtosis(Y)


%******************************正态性检验***********************************
[h,p] = jbtest(X)
[h,p] = kstest(X,[X,normcdf(X,mean(X),std(X))])
[h, p] = lillietest(X)

[h,p] = jbtest(Y)
[h,p] = kstest(Y,[Y,normcdf(Y,mean(Y),std(Y))])
[h, p] = lillietest(Y)


%****************************求经验分布函数值*******************************
[fx, Xsort] = ecdf(X);
[fy, Ysort] = ecdf(Y);
U1 = spline(Xsort(2:end),fx(2:end),X);
V1 = spline(Ysort(2:end),fy(2:end),Y);


%*******************************核分布估计**********************************
U2 = ksdensity(X,X,'function','cdf');
V2 = ksdensity(Y,Y,'function','cdf');


% **********************绘制经验分布函数图和核分布估计图**********************
[Xsort,id] = sort(X);
figure;
plot(Xsort,U1(id),'c','LineWidth',5);
hold on
plot(Xsort,U2(id),'k-.','LineWidth',2);
legend('经验分布函数','核分布估计', 'Location','NorthWest');
xlabel('沪市日收益率');
ylabel('F(x)');

[Ysort,id] = sort(Y);
figure;
plot(Ysort,V1(id),'c','LineWidth',5);
hold on
plot(Ysort,V2(id),'k-.','LineWidth',2);
legend('经验分布函数','核分布估计', 'Location','NorthWest');
xlabel('深市日收益率');
ylabel('F(x)');


%****************************绘制二元频数直方图*****************************
U = ksdensity(X,X,'function','cdf');
V = ksdensity(Y,Y,'function','cdf');
figure;
hist3([U(:) V(:)],[30,30])
xlabel('U（沪市）');
ylabel('V（深市）');
zlabel('频数');


%****************************绘制二元频率直方图*****************************
figure;
hist3([U(:) V(:)],[30,30])
h = get(gca, 'Children');
cuv = get(h, 'ZData');
set(h,'ZData',cuv*30*30/length(X));
xlabel('U（沪市）');
ylabel('V（深市）');
zlabel('c(u,v)');


%***********************求Copula中参数的估计值******************************
rho_norm = copulafit('Gaussian',[U(:), V(:)])
[rho_t,nuhat,nuci] = copulafit('t',[U(:), V(:)])


%********************绘制Copula的密度函数和分布函数图************************
[Udata,Vdata] = meshgrid(linspace(0,1,31));
Cpdf_norm = copulapdf('Gaussian',[Udata(:), Vdata(:)],rho_norm);
Ccdf_norm = copulacdf('Gaussian',[Udata(:), Vdata(:)],rho_norm);
Cpdf_t = copulapdf('t',[Udata(:), Vdata(:)],rho_t,nuhat);
Ccdf_t = copulacdf('t',[Udata(:), Vdata(:)],rho_t,nuhat);

% 绘制二元正态Copula的密度函数和分布函数图
figure;
surf(Udata,Vdata,reshape(Cpdf_norm,size(Udata)));
xlabel('U');
ylabel('V');
zlabel('c(u,v)');
figure;
surf(Udata,Vdata,reshape(Ccdf_norm,size(Udata)));
xlabel('U');
ylabel('V');
zlabel('C(u,v)');

% 绘制二元t-Copula的密度函数和分布函数图
figure;
surf(Udata,Vdata,reshape(Cpdf_t,size(Udata)));
xlabel('U');
ylabel('V');
zlabel('c(u,v)');
figure;
surf(Udata,Vdata,reshape(Ccdf_t,size(Udata)));
xlabel('U');
ylabel('V');
zlabel('C(u,v)');


%**************求Kendall秩相关系数和Spearman秩相关系数***********************
Kendall_norm = copulastat('Gaussian',rho_norm)
Spearman_norm = copulastat('Gaussian',rho_norm,'type','Spearman')
Kendall_t = copulastat('t',rho_t)
Spearman_t = copulastat('t',rho_t,'type','Spearman')
% MATLAB R2014a版本新用法
% Spearman_t = copulastat('t',rho_t,nuhat,'type','Spearman')

Kendall = corr([X,Y],'type','Kendall')
Spearman = corr([X,Y],'type','Spearman')


%******************************模型评价*************************************
[fx, Xsort] = ecdf(X);
[fy, Ysort] = ecdf(Y);
U = spline(Xsort(2:end),fx(2:end),X);
V = spline(Ysort(2:end),fy(2:end),Y);
C = @(u,v)mean((U <= u).*(V <= v));
[Udata,Vdata] = meshgrid(linspace(0,1,31));
for i=1:numel(Udata)
    CopulaEmpirical(i) = C(Udata(i),Vdata(i));
end

figure;
% 绘制经验Copula分布函数图像
surf(Udata,Vdata,reshape(CopulaEmpirical,size(Udata)))
xlabel('U');
ylabel('V');
zlabel('Empirical Copula C(u,v)');

% 通过循环计算经验Copula函数在原始样本点处的函数值
CUV = zeros(size(U(:)));
for i=1:numel(U)
    CUV(i) = C(U(i),V(i));
end

% 计算线性相关参数为0.9264的二元正态Copula函数在原始样本点处的函数值
rho_norm = 0.9264;
Cgau = copulacdf('Gaussian',[U(:), V(:)],rho_norm);
% 计算线性相关参数为0.9325，自由度为4的二元t-Copula函数在原始样本点处的函数值
rho_t = 0.9325;
k = 4.0089;
Ct = copulacdf('t',[U(:), V(:)],rho_t,k);
% 计算平方欧氏距离
dgau2 = (CUV-Cgau)'*(CUV-Cgau)
dt2 = (CUV-Ct)'*(CUV-Ct)
