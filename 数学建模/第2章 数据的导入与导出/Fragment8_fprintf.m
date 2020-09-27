%--------------------------------------------------------------------------
%                    调用fprintf函数写数据到文件或屏幕
%--------------------------------------------------------------------------
% CopyRight：xiezhh

y = fprintf(1, '祝福我们伟大的新中国%d周岁生日快乐！！！', 60)

x = 10*rand(8,5);
fid = fopen('examp02_01.txt','wt');
% 把矩阵x以指定格式写入文件examp02_01.txt
fprintf(fid,'%-f    %-f    %-f    %-f    %-f    %-f    %-f    %-f\n', x);
fclose(fid);

x = rand(6,5)/10000;
fid = fopen('examp02_02.txt','wt');
% 把矩阵x以指定格式写入文件examp02_02.txt
fprintf(fid,'%-e  %-e  %-e  %-e  %-e  %-e\n', x);
fclose(fid);

x=10*rand(9,4);
fid = fopen('examp02_03.txt','wt');
% 把矩阵x以指定格式写入文件examp02_03.txt
fprintf(fid,'%f,%f,%f,%f,%f,%f,%f,%f,%f\n',x);
fclose(fid);

x = 10*rand(5,4);
fid = fopen('examp02_04.txt','wt');
% 把矩阵x以指定格式写入文件examp02_04.txt
fprintf(fid,'%-f    %-f,    %-f;    %-f*    %-f\n',x);
fclose(fid);

w = 10*rand(1,4);
x = 10*rand(1,3);
y = 10*rand(1,2);
z = 10*rand;
fid = fopen('examp02_05.txt','at');
% 把向量w,x,y,z分别以指定格式写入文件examp02_05.txt
fprintf(fid,'%-f    %-f    %-f    %-f\n', w);
fprintf(fid,'%-f    %-f    %-f\n', x);
fprintf(fid,'%-f    %-f\n', y);
fprintf(fid,'%-f\n', z);
fclose(fid);

x = 10*rand(1,3);
y = 10*rand(1,5);
z = 10*rand(1,4);
fid = fopen('examp02_06.txt','at');
% 把向量x,y,z分别以指定格式写入文件examp02_06.txt
fprintf(fid,'%-f    %-f    %-f\n', x);
fprintf(fid,'%-f    %-f    %-f    %-f    %-f\n', y);
fprintf(fid,'%-f    %-f    %-f    %-f\n', z);
fclose(fid);

x = 10*rand(6,3);
fid = fopen('examp02_07.txt','at');
fprintf(fid,'这是%d行头文件，\n你可以选择跳过，读取后面的数据。\n', 2);
% 把矩阵x以指定格式写入文件examp02_07.txt
fprintf(fid,'%-f,    %-f,    %-f,    %-f,    %-f,    %f\n', x);
fclose(fid);

x = 10*rand(6,3);
y = 10*rand(3,2);
fid = fopen('examp02_08.txt','at');
fprintf(fid,'这是%d行头文件，\n你可以选择跳过，读取后面的数据。\n', 2);
fprintf(fid,'%-f    %-f    %-f    %-f    %-f    %f\n', x);
fprintf(fid,'这里还有两行文字说明和两行数据，\n看你还有没有办法！\n');
% 把矩阵y以指定格式写入文件examp02_08.txt
fprintf(fid,'%-f    %-f    %-f    %-f    %-f    %f\n', y);
fclose(fid);

x = 10*rand(2,12);
fid = fopen('examp02_09.txt','wt');
% 把矩阵x以指定格式写入文件examp02_09.txt
fprintf(fid,'%f+%fi, %f+%fi, %f+%fi, %f+%fi\n', x);
fclose(fid);

dt = [2009 08 19 10 39 56.171
         2009 08 20 10 39 56.171
         2009 08 21 10 39 56.171
         2009 08 22 10 39 56.171]';
fid = fopen('examp02_10.txt','wt');
% 把矩阵dt以指定格式写入文件examp02_10.txt
fprintf(fid,'%d-%d-%d,  %d:%d:%5.3f AM\n', dt);  
fclose(fid);

x = ['xiezh'; 'molih'; 'liaoj'; 'lijun'; 'xiagk'];
y = [18 16 15 20 15]';
z = [170 160 160 175 172]';
w = [65 52 50 70 56]';
fid = fopen('examp02_11.txt','at');
fm = 'Name: %s Age: %d Height: %d Weight: %d kg\n';
% 通过循环将x,y,z和w按指定格式写入文件examp02_11.txt
for i = 1:5    
       fprintf(fid, fm, x(i,:),y(i),z(i),w(i));
   end
fclose(fid);


x = [1 2 3; 4 5 6; 7 8 9; 10 11 12]
% 把矩阵x以指定格式显示在屏幕上
fprintf(1, '    %-d    %-d    %-d    %d\n', x);
