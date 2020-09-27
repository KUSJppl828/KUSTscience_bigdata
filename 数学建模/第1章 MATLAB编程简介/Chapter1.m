%--------------------------------------------------------------------------
%  第1章   MATLAB编程简介
%--------------------------------------------------------------------------
% CopyRight：xiezhh

%% examp1.2-1
x = 1
y = 1+2+sqrt(9)
z = 'Hellow World !!!'

N = namelengthmax
abceddddddddddddddddddddddddddddddddddddddddddddddddddddddddwertyu = 1

(7189+(1021-913)*80)/64^0.5

pi
pi = 1
clear pi
pi

pi = 1;
clear = 2;
clear pi

iskeyword('for')
iskeyword('xiezhh')

whos

%% examp1.3-1
x = [1  -1.65  2.2  -3.1];
y1 = abs(x)
y2 = sin(x)
y3 = round(x)
y4 = floor(x)
y5 = ceil(x)
y6 = min(x)
y7 = mean(x)
y8 = range(x)
y9 = sign(x)

%% examp1.4-1
x = []

%% examp1.4-2
y = [1, 2, 3;4  5  6;7  8, 9]
z = [1  2  3
     4  5  6
     7  8  9]

%% examp1.4-3
x = 1:10 
y = 1:2:10
z = [1:3; 4:6; 7:9]

%% examp1.4-4
x = linspace(1, 10, 10)

%% examp1.4-5
x = [1  2  3; 4  5  6]
size(x)
[m, n] = size(x)

%% examp1.4-6
x = [1  2  3; 4  5  6; 7  8  9]
y1 = x(1, 2)
y2 = x(2:3, 1:2)
y3 = x(:, 1:2)
y4 = x(1, :)
y5 = x(:)'
y6 = x(3:6)

%% examp1.4-7
x1 = [1  2  3];
x2 = [4  5  6];
x = [x1; x2]
y = reshape(x, [3, 2])
z = repmat(x, [2, 2])

%% examp1.4-8
x = ['abc'; 'def'; 'ghi'] 
size(x)

%% examp1.4-9
x = 2i+5
y = [1  2  3; 4  5  6]*i+7
a = [1  2; 3  4];
b = [5  6; 7  8];
c = complex(a,b)

%% examp1.4-10
syms a b c d
x = [a  b; c  d]
y = [1  2  3; 4  5  6];  
y = sym(y)

%% examp1.4-11
A = zeros(3)
B = ones(3,5)
C = eye(3,5)
D = diag([1 2 3])
E = diag(D)
F = rand(3)
G = magic(3)

%% examp1.4-12
% 定义一个2行，2列，2页的3维数组
x(1:2, 1:2, 1)=[1  2; 3  4];
x(1:2, 1:2, 2)=[5  6; 7  8];

%% examp1.4-13
A1 = [1  2; 3  4];
A2 = [5  6; 7  8];
A = cat(3, A1, A2)

%% examp1.4-14
x = reshape(1:12, [2, 2, 3]) 

%% examp1.4-15
x = repmat([1  2; 3  4], [1 1 2])

%% examp1.4-16
c1 = {[1  2; 3  4], 'xiezhh', 10; [5  6  7], ['abc';'def'], 'I LOVE MATLAB'}

%% examp1.4-17
c2 = cell(2,4)
c2{2, 3} = [1  2  3]

%% examp1.4-18
c = {[1  2], 'xie', 'xiezhh'; 'MATLAB', [3  4; 5  6], 'I LOVE MATLAB'}
c(2, 2)
c{2, 2}

c = {[1  2],  'xiezhh'; 'MATLAB', [3  4; 5  6]};
celldisp(c)

%% examp1.4-19
% 通过直接赋值方式定义一个1行2列的结构体数组
struct1(1).name = 'xiezhh';
struct1(2).name = 'heping';
struct1(1).age = 31;
struct1(2).age = 22;
struct1 

%% examp1.4-20
struct2 = struct('name', {'xiezhh', 'heping'}, 'age',{31, 22})
struct2(1).name

%% examp1.4-21
A1 = rand(60,50);
B1 = mat2cell(A1, [10 20 30], [25 25])
C1 = cell2mat(B1);
isequal(A1,C1)
A2 = [1 2 3 4;5 6 7 8;9 10 11 12];
B2 = num2cell(A2)
C = {'Heping', 'Tianjin', 22;  'Xiezhh', 'Xingyang', 31}
fields = {'Name', 'Address', 'Age'};
S = cell2struct(C, fields, 2)
CS = struct2cell(S)
isequal(C,CS')
x = [1;2;3;4;5];
x = cellstr(num2str(x));
y = strcat('xiezhh', x, '.txt')

%% examp1.4-22
A = [1  2; 3  4];
B = [5  6; 7  8];
C = A+B
D = A-B

%% examp1.4-23
A = [1  2  3; 4  5  6];
B = [1  1  1  1; 2  2  2  2; 3  3  3  3];
C = A*B 
D = [1  1  1; 2  2  2];
E = A.*D 

%% examp1.4-24
A = [2  3  8; 1  -2  -4; -5  3  1];
b = [-5; 3; 2];
x = A\b
B = A;
C = A./B

%% examp1.4-25
A = [1  2; 3  4];
B = A ^ 2
C = A .^ 2
D = A .^ A

%% examp1.4-26
A = [1  2; 3  4];
B = [2  2; 2  2];
C1 = A > B 
C2 = A ~= B
C3 = A >=2

%% examp1.4-27
A = [0  0  1  2];
B = [0  -2  0  1];
C1 = A | B
C2 = A & B
C3 = ~ A
C4 = xor(A, B)
x = 5;
y = 0;
x || y
x && y

%% examp1.4-28
A = [1  2  3; 4  5  6; 7  8  9]
B = A'

%% examp1.4-29
A = [1  2  3; 4  5  6; 7  8  9];
B1 = flipud(A)
B2 = fliplr(A)
B3 = rot90(A)

%% examp1.4-30
A = [1  2; 3  4];
d1 = det(A)
syms a b c d
B = [a  b; c  d];
d2 = det(B)

%% examp1.4-31
A = [1  2; 3  4];
Ai = inv(A)
syms a b c d
B = [a  b; c  d];
Bi = inv(B) 
C = [1  2  3; 4  5  6];
Cpi = pinv(C)
D = C * Cpi * C

%% examp1.4-32
A = [5  0  4; 3  1  6; 0  2  3];
d = eig(A)
[V, D] = eig(A)
[Vs, Ds] = eig(sym(A))

%% examp1.4-33
A = [1  2  3; 4  5  6; 7  8  9];
t = trace(A)
r = rank(A)

%% examp1.5-1
A = input('请输入三角形的三条边：');
if A(1) + A(2) > A(3) & A(1) + A(3) > A(2) & A(2) + A(3) > A(1)
    p = (A(1) + A(2) + A(3)) / 2;
    s = sqrt(p*(p - A(1))*(p - A(2))*(p - A(3)));
    disp(['该三角形面积为：' num2str(s)]);
else
    disp('不能构成一个三角形。')
end

%% examp1.5-2
num = input('请输入一个数：');
switch num
    case -1
        disp('I am a teacher.');
    case 0
        disp('I am a student.');
    case 1
        disp('You are a teacher.');
    otherwise
        disp('You are a student.');
end

%% examp1.5-3
% 程序1：for循环
y = 0;
for i = 1:inf 
    y = y + i^2;
    if  y >= 2000
        break;
    end
end
n = i - 1 
y = y - i^2

% 程序2：while循环
y = 0;
i = 0;
while  y < 2000
    i = i + 1;
    y = y + i^2;
end
n = i-1 
y = y-i^2

%% examp1.5-4
A = [1,2,3;4,5,6]; B = [7,8,9;10,11,12];
try
    X = A*B
catch ME
    disp(ME.message);    % 显示出错原因
end

%% examp1.5-5
A = ones(100,100,100);
[m, n, p] = size(A);
tic    % 开始计时
SA1 = sum(A(:))
t1 = toc    % 返回sum函数所用时间
tic
SA2 = 0;

for i = 1 : m
    for j = 1 : n
        for k = 1 : p
            SA2 = SA2 + A(i,j,k);
        end
    end
end
SA2
t2 = toc

%% examp1.5-6
A = ones(100,100,100);
[m, n, p] = size(A);
tic
A2_1 = A.^2;
t1 = toc
tic
for i = 1 : m
    for j = 1 : n
        for k = 1 : p
            A2_2(i, j, k) = A(i,j,k)^2;
        end
    end
end
t2 = toc

%% examp1.5-7
A = ones(100,100,100);
[m, n, p] = size(A);
A2_1 = A;    %预定义数组A2_1
tic
for i = 1 : m
    for j = 1 : n
        for k = 1 : p
            A2_1(i, j, k) = A(i,j,k)^2;
        end
    end
end
t1 = toc   % 返回有预定义的运算时间
tic
for i = 1 : m
    for j = 1 : n
        for k = 1 : p
            A2_2(i, j, k) = A(i,j,k)^2;   %数组A2_2没有预定义
        end
    end
end
t2 = toc

%% examp1.6-2
fun1 = @(x,y) cos(x).*sin(y)
x = [0,1,2];
y = [-1,0,1];
z = fun1(x,y)

%% examp1.6-3
fun2 = @(x)(x>=-1 & x<0).*sin(pi*x.^2)+(x>=0).*exp(1-x);
fun2([-0.5,0,0.5])

%% examp1.6-4
fun3 = inline('cos(x).*sin(y)','x','y')
x = [0,1,2];
y = [-1,0,1];
z = fun3(x,y)

%% examp1.7-01
h = line([0 1],[0 1]) 
get(h)

%% examp1.7-02
subplot(1, 2, 1); 
h1 = line([0 1],[0 1]) ;
text(0, 0.5, '未改变线宽') ;
subplot(1, 2, 2);
h2 = line([0 1],[0 1]) ;
set(h2, 'LineWidth', 3)
text(0, 0.5, '已改变线宽') ;

%% examp1.7-1
x = 0 : 0.25 : 2*pi;
y = sin(x);
plot(x, y, '-ro',...
              'LineWidth',2,...
              'MarkerEdgeColor','k',...
              'MarkerFaceColor',[0.49,  1,  0.63],...
              'MarkerSize',12)
xlabel('X');
ylabel('Y'); 

%% examp1.7-2
t = linspace(0,2*pi,60);
x = cos(t);
y = sin(t);
plot(t,x,':','LineWidth',2);
hold on;
plot(t,y,'r-.','LineWidth',3);
plot(x,y,'k','LineWidth',2.5);
axis equal;
xlabel('X');
ylabel('Y');
legend('x = cos(t)','y = sin(t)','单位圆','Location','NorthEast');

%% examp1.7-3
P = [3 1; 1 4]; 
r = 5;
[V, D] = eig(P);
a = sqrt(r/D(1));
b = sqrt(r/D(4));
t = linspace(0, 2*pi, 60);
xy = V*[a*cos(t); b*sin(t)];
plot(xy(1,:),xy(2,:), 'k', 'linewidth', 3);
h = annotation('textarrow',[0.606 0.65],[0.55 0.65]);
set(h, 'string','3x^2+2xy+4y^2 = 5', 'fontsize', 15);
h = title('这是一个椭圆曲线', 'fontsize', 18, 'fontweight', 'bold');
set(h, 'position', [-0.00345622 1.35769 1.00011]);
axis([-1.5 1.5 -1.2 1.7]);
xlabel('X');
ylabel('Y');

%% examp1.7-4
a = [-19.6749   22.2118    5.0905];
x = 0:0.01:1;
y = a(1)+a(2)/2*(x-0.17).^2+a(3)/4*(x-0.17).^4;
plot(x,y);
xlabel('X');
ylabel('Y = f(X)');
text('Interpreter','latex',...
	'String',['$$-19.6749+\frac{22.2118}{2}(x-0.17)^2'...
                '+\frac{5.0905}{4}(x-0.17)^4$$'],'Position',[0.05, -12],...
	'FontSize',12);

%% examp1.7-5
x = linspace(0,2*pi,60);
y = sin(x);
h = plot(x,y);
grid on;
set(h,'Color','k','LineWidth',2);
XTickLabel = {'0','pi/2','pi','3pi/2','2pi'};
set(gca,'XTick',[0:pi/2:2*pi],...
           'XTickLabel',XTickLabel,...
           'TickDir','out');
xlabel('0 \leq \Theta \leq 2\pi');
ylabel('sin(\Theta)'); 
text(8*pi/9,sin(8*pi/9),'\leftarrow sin(8\pi \div 9)',...
        'HorizontalAlignment','left')
axis([0 2*pi -1 1]);

%% examp1.7-6
x = normrnd(0, 1, 1000, 1);
hist(x, 20);
xlabel('样本数据');
ylabel('频数') ;
figure;
cdfplot(x);

%% examp1.7-7
subplot(3, 3, 1);
f = @(x)200*sin(x)./x;
fplot(f, [-20 20]);
title('y = 200*sin(x)/x');

subplot(3, 3, 2);
ezplot('x^2 + y^2 = 1', [-1.1 1.1]);
axis equal;
title('单位圆');

subplot(3, 3, 3);
ezpolar('1+cos(t)');
title('心形图');

subplot(3, 3, 4);
x = [10  10  20  25  35];
name = {'赵', '钱', '孙', '李', '谢'};
explode = [0 0 0 0 1];
pie(x, explode, name)
title('饼图');

subplot(3, 3, 5);
stairs(-2*pi:0.5:2*pi,sin(-2*pi:0.5:2*pi)); 
title('楼梯图');

subplot(3, 3, 6);
stem(-2*pi:0.5:2*pi,sin(-2*pi:0.5:2*pi));
title('火柴杆图');

subplot(3, 3, 7);
Z = eig(randn(20,20));
compass(Z); 
title('罗盘图');

subplot(3, 3, 8); 
theta = (-90:10:90)*pi/180; 
r = 2*ones(size(theta));
[u,v] = pol2cart(theta,r);
feather(u,v);
title('羽毛图');

subplot(3, 3, 9); 
t = (1/16:1/8:1)'*2*pi;
fill(sin(t), cos(t),'r');
axis square;   title('八边形');

%% examp1.7-8
t = linspace(0, 10*pi, 300);
plot3(20*sin(t), 20*cos(t), t, 'r', 'linewidth', 2);
hold on
quiver3(0,0,0,1,0,0,25,'k','filled','LineWidth',2);
quiver3(0,0,0,0,1,0,25,'k','filled','LineWidth',2);
quiver3(0,0,0,0,0,1,40,'k','filled','LineWidth',2);
grid on
xlabel('X'); ylabel('Y'); zlabel('Z');
axis([-25 25 -25 25 0 40]); 
view(-210,30);

%% examp1.7-9
[x,y] = meshgrid(1:4, 2:5)
plot(x, y, 'r',x', y', 'r', x, y, 'k.','markersize',18);
axis([0 5 1 6]);
xlabel('X');  ylabel('Y');

%% examp1.7-10
t = linspace(-pi,pi,20);
[X, Y] = meshgrid(t);
Z = cos(X).*sin(Y);

subplot(2, 2, 1);
mesh(X, Y, Z); 
title('mesh');

subplot(2, 2, 2);
surf(X, Y, Z);
alpha(0.5);
title('surf'); 

subplot(2, 2, 3);
surfl(X, Y, Z);
title('surfl');

subplot(2, 2, 4);
surfc(X, Y, Z);
title('surfc'); 

%% examp1.7-11
[X,Y] = meshgrid(-2:.2:2);
Z = X.*exp(-X.^2 - Y.^2);
[DX,DY] = gradient(Z,0.2,0.2);
contour(X,Y,Z);
hold on;
quiver(X,Y,DX,DY);
h = get(gca,'Children');
set(h, 'Color','k');

%% examp1.7-12
% 绘制圆柱面
subplot(2,2,1);
[x,y,z] = cylinder;
surf(x,y,z);
title('圆柱面')

% 绘制哑铃面
subplot(2,2,2);
t = 0:pi/10:2*pi;
[X,Y,Z] = cylinder(2+cos(t));
surf(X,Y,Z);
title('哑铃面')

% 绘制球面，半径为10，球心 (1,1,1)
subplot(2,2,3); 
[x,y,z] = sphere;
surf(10*x+1,10*y+1,10*z+1);
axis equal;
title('球面') 

% 绘制椭球面
subplot(2,2,4);
a=4;
b=3;
t = -b:b/10:b;
[x,y,z] = cylinder(a*sqrt(1-t.^2/b^2),30);
surf(x,y,z);
title('椭球面')

%% examp1.7-13
% 调用ezsurf函数绘制参数方程形式的螺旋面,并设置参数取值范围
ezsurf('u*sin(v)','u*cos(v)', '4*v',[-2*pi,2*pi,-2*pi,2*pi])
axis([-7 7 -7 7 -30 30]);

%% examp1.7-14
% 饼图
subplot(2,3,1);
pie3([2347,1827,2043,3025]);
title('三维饼图');

% 柱状图
subplot(2,3,2);
bar3(magic(4));
title('三维柱状图');

% 火柴杆图
subplot(2,3,3);
y=2*sin(0:pi/10:2*pi);
stem3(y);
title('三维火柴杆图');

% 填充图
subplot(2,3,4);
fill3(rand(3,5),rand(3,5),rand(3,5), 'y' );
title('三维填充图');

% 三维向量场图
subplot(2,3,5); 
[X,Y] = meshgrid(0:0.25:4,-2:0.25:2);
Z = sin(X).*cos(Y);
[Nx,Ny,Nz] = surfnorm(X,Y,Z);
surf(X,Y,Z);
hold on;
quiver3(X,Y,Z,Nx,Ny,Nz,0.5);
title('三维向量场图');
axis([0 4 -2 2 -1 1]);

% 立体切片图（四维图）
subplot(2,3,6);
t = linspace(-2,2,20);
[X,Y,Z] = meshgrid(t,t,t);
V = X.*exp(-X.^2-Y.^2-Z.^2);    
xslice = [-1.2,0.8,2];
yslice = 2;
zslice = [-2,0];
slice(X,Y,Z,V,xslice,yslice,zslice);
title('立体切片图（四维图）');

%% examp1.7-15
figure;
[X,Y,Z] = sphere;
surf(X,Y,Z); 
colormap(lines);
shading interp
hold on; 
mesh(2*X,2*Y,2*Z)
hidden off 
axis equal 
axis off

figure; 
surf(X,Y,Z,'FaceColor','r'); 
hold on; 
surf(2*X,2*Y,2*Z,'FaceAlpha',0.4);
axis equal
axis off

%% examp1.7-16
t=0:pi/20:2*pi;
[x,y,z]= cylinder(2+sin(t),100);
surf(x,y,z);
xlabel('X'); ylabel('Y'); zlabel('Z');
set(gca,'color','none');
set(gca,'XColor',[0.5 0.5 0.5]);
set(gca,'YColor',[0.5 0.5 0.5]);
set(gca,'ZColor',[0.5 0.5 0.5]);
shading interp;
colormap(copper);
light('Posi',[-4 -1 0]); 
lighting phong;
material metal; 
hold on;
plot3(-4,-1,0,'p','markersize', 18);
text(-4,-1,0,'光源','fontsize',14,'fontweight','bold');

%% examp1.7-17
vert = [0 0 0;0 200 0;200 200 0;200 0 0;0 0 100;...
           0 200 100;200 200 100;200 0 100];
fac = [1 2 3 4;2 6 7 3;4 3 7 8;1 5 8 4;1 2 6 5;5 6 7 8];
view(3);
h = patch('faces',fac,'vertices',vert,'FaceColor','g');
set(h,'FaceAlpha',0.25);
hold on;
[x0,y0,z0] = sphere;
x = 30 + 30*x0; y = 50 + 30*y0; z = 50 + 30*z0;
h1 = surf(x,y,z,'linestyle','none','FaceColor','r','EdgeColor','none');
x = 110 + 30*x0; y = 110 + 30*y0; z = 50 + 30*z0;
h2 = surf(x,y,z,'linestyle','none','FaceColor','b','EdgeColor','none');
x = 110 + 30*x0; y = 30 + 30*y0; z = 50 + 30*z0;
h3 = surf(x,y,z,'linestyle','none','FaceColor','y','EdgeColor','none');
lightangle(45,30);
lighting phong;
axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');

%% examp1.7-18
x = 0 : 0.25 : 2*pi; 
y = sin(x); 
plot(x, y);
hgexport(gcf,'-clipboard');

%% examp1.7-19
x = 0 : 0.25 : 2*pi; 
y = sin(x); 
h = plot(x, y);
saveas(h,'xiezhh.jpg');

%% examp1.7-20
x = 0 : 0.25 : 2*pi;
y = sin(x);
h = plot(x, y);
print; 
print -dmeta
print -djpeg heping.jpg