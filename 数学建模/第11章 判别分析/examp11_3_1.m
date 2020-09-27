%--------------------------------------------------------------------------
%  examp11.3-1  加载fisheriris.mat中数据，进行贝叶斯判别
%--------------------------------------------------------------------------
% CopyRight：xiezhh

%********************************加载数据***********************************
load fisheriris


%**********************************查看数据*********************************
head0 = {'Obj', 'x1', 'x2', 'x3', 'x4', 'Class'};
[head0; num2cell([[1:150]', meas]), species]


%*********************************贝叶斯判别********************************
ObjBayes = NaiveBayes.fit(meas, species);
pre0 = ObjBayes.predict(meas);
[CLMat, order] = confusionmat(species, pre0);
[[{'From/To'},order'];order, num2cell(CLMat)]


% 查看误判样品编号
gindex1 = grp2idx(pre0);
gindex2 = grp2idx(species);
errid = find(gindex1 ~= gindex2)

% 查看误判样品的误判情况
head1 = {'Obj', 'From', 'To'};
[head1; num2cell(errid), species(errid), pre0(errid)]


% 对未知类别样品进行判别
x = [5.8	2.7	1.8	0.73
    5.6	3.1	3.8	1.8
    6.1	2.5	4.7	1.1
    6.1	2.6	5.7	1.9
    5.1	3.1	6.5	0.62
    5.8	3.7	3.9	0.13
    5.7	2.7	1.1	0.12
    6.4	3.2	2.4	1.6
    6.7	3	1.9	1.1
    6.8	3.5	7.9	1
    ];
pre1 = ObjBayes.predict(x)

