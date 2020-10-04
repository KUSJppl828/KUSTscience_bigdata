# 导入第三方包
import pandas as pd
import numpy as np
import seaborn as sns

# 数据读取
income = pd.read_excel(r'C:\Users\Administrator\Desktop\income.xlsx')

# 查看数据集是否存在缺失值
income.apply(lambda x:np.sum(x.isnull()))

# 缺失值处理
income.fillna(value = {'workclass':income.workclass.mode()[0],
                              'occupation':income.occupation.mode()[0],
                              'native-country':income['native-country'].mode()[0]}, inplace = True)
income.head()


# 数据的探索性分析
income.describe()

income.describe(include =[ 'object'])


# 导入绘图模块
import matplotlib.pyplot as plt
# 设置绘图风格
plt.style.use('ggplot')
# 设置多图形的组合
fig, axes = plt.subplots(2, 1)
# 绘制不同收入水平下的年龄核密度图
income.age[income.income == ' <=50K'].plot(kind = 'kde', label = '<=50K', ax = axes[0], legend = True, linestyle = '-')
income.age[income.income == ' >50K'].plot(kind = 'kde', label = '>50K', ax = axes[0], legend = True, linestyle = '--')
# 绘制不同收入水平下的周工作小时数和密度图
income['hours-per-week'][income.income == ' <=50K'].plot(kind = 'kde', label = '<=50K', ax = axes[1], legend = True, linestyle = '-')
income['hours-per-week'][income.income == ' >50K'].plot(kind = 'kde', label = '>50K', ax = axes[1], legend = True, linestyle = '--')
# 显示图形
plt.show()


# 构造不同收入水平下各种族人数的数据
race = pd.DataFrame(income.groupby(by = ['race','income']).aggregate(np.size).loc[:,'age'])
# 重设行索引
race = race.reset_index()
# 变量重命名
race.rename(columns={'age':'counts'}, inplace=True)
# 排序
race.sort_values(by = ['race','counts'], ascending=False, inplace=True)

# 构造不同收入水平下各家庭关系人数的数据
relationship = pd.DataFrame(income.groupby(by = ['relationship','income']).aggregate(np.size).loc[:,'age'])
relationship = relationship.reset_index()
relationship.rename(columns={'age':'counts'}, inplace=True)
relationship.sort_values(by = ['relationship','counts'], ascending=False, inplace=True)

# 设置图框比例，并绘图
plt.figure(figsize=(9,5))
sns.barplot(x="race", y="counts", hue = 'income', data=race)
plt.show()

plt.figure(figsize=(9,5))
sns.barplot(x="relationship", y="counts", hue = 'income', data=relationship)
plt.show()


# 离散变量的重编码
for feature in income.columns:
    if income[feature].dtype == 'object':
        income[feature] = pd.Categorical(income[feature]).codes
income.head()

# 删除变量
income.drop(['education','fnlwgt'], axis = 1, inplace = True)
income.head()


# 数据拆分
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(income.loc[:,'age':'native-country'], 
                                                    income['income'], train_size = 0.75, 
                                                    random_state = 1234)
print('训练数据集共有%d条观测' %X_train.shape[0])
print('测试数据集共有%d条观测' %X_test.shape[0])


# 导入k近邻模型的类
from sklearn.neighbors import KNeighborsClassifier
# 构建k近邻模型
kn = KNeighborsClassifier()
kn.fit(X_train, y_train)
print(kn)

# 预测测试集
kn_pred = kn.predict(X_test)
print(pd.crosstab(kn_pred, y_test))

# 模型得分
print('模型在训练集上的准确率%f' %kn.score(X_train,y_train))
print('模型在测试集上的准确率%f' %kn.score(X_test,y_test))

# 导入模型评估模块
from sklearn import metrics

# 计算ROC曲线的x轴和y轴数据
fpr, tpr, _ = metrics.roc_curve(y_test, kn.predict_proba(X_test)[:,1])
# 绘制ROC曲线
plt.plot(fpr, tpr, linestyle = 'solid', color = 'red')
# 添加阴影
plt.stackplot(fpr, tpr, color = 'steelblue')
# 绘制参考线
plt.plot([0,1],[0,1], linestyle = 'dashed', color = 'black')
# 往图中添加文本
plt.text(0.6,0.4,'AUC=%.3f' % metrics.auc(fpr,tpr), fontdict = dict(size = 18))
plt.show()


# 导入GBDT模型的类
from sklearn.ensemble import GradientBoostingClassifier
# 构建GBDT模型
gbdt = GradientBoostingClassifier()
gbdt.fit(X_train, y_train)
print(gbdt)

# 预测测试集
gbdt_pred = gbdt.predict(X_test)
print(pd.crosstab(gbdt_pred, y_test))

# 模型得分
print('模型在训练集上的准确率%f' %gbdt.score(X_train,y_train))
print('模型在测试集上的准确率%f' %gbdt.score(X_test,y_test))

# 绘制ROC曲线
fpr, tpr, _ = metrics.roc_curve(y_test, gbdt.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, linestyle = 'solid', color = 'red')
plt.stackplot(fpr, tpr, color = 'steelblue')
plt.plot([0,1],[0,1], linestyle = 'dashed', color = 'black')
plt.text(0.6,0.4,'AUC=%.3f' % metrics.auc(fpr,tpr), fontdict = dict(size = 18))
plt.show()



# K近邻模型的网格搜索法
# 导入网格搜索法的函数
from sklearn.grid_search import GridSearchCV
# 选择不同的参数
k_options = list(range(1,12))
parameters = {'n_neighbors':k_options}
# 搜索不同的K值
grid_kn = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = parameters, cv=10, scoring='accuracy', verbose=0)
grid_kn.fit(X_train, y_train)
print(grid_kn)
# 结果输出
grid_kn.grid_scores_, grid_kn.best_params_, grid_kn.best_score_  

# 预测测试集
grid_kn_pred = grid_kn.predict(X_test)
print(pd.crosstab(grid_kn_pred, y_test))

# 模型得分
print('模型在训练集上的准确率%f' %grid_kn.score(X_train,y_train))
print('模型在测试集上的准确率%f' %grid_kn.score(X_test,y_test))

# 绘制ROC曲线
fpr, tpr, _ = metrics.roc_curve(y_test, grid_kn.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, linestyle = 'solid', color = 'red')
plt.stackplot(fpr, tpr, color = 'steelblue')
plt.plot([0,1],[0,1], linestyle = 'dashed', color = 'black')
plt.text(0.6,0.4,'AUC=%.3f' % metrics.auc(fpr,tpr), fontdict = dict(size = 18))
plt.show()



# GBDT模型的网格搜索法
# 选择不同的参数
learning_rate_options = [0.01,0.05,0.1]
max_depth_options = [3,5,7,9]
n_estimators_options = [100,300,500]
parameters = {'learning_rate':learning_rate_options,'max_depth':max_depth_options,'n_estimators':n_estimators_options}

grid_gbdt = GridSearchCV(estimator = GradientBoostingClassifier(), param_grid = parameters, cv=10, scoring='accuracy')
grid_gbdt.fit(X_train, y_train)

# 结果输出
grid_gbdt.grid_scores_, grid_gbdt.best_params_, grid_gbdt.best_score_  


# 预测测试集
grid_gbdt_pred = grid_gbdt.predict(X_test)
print(pd.crosstab(grid_gbdt_pred, y_test))

# 模型得分
print('模型在训练集上的准确率%f' %grid_gbdt.score(X_train,y_train))
print('模型在测试集上的准确率%f' %grid_gbdt.score(X_test,y_test))

# 绘制ROC曲线
fpr, tpr, _ = metrics.roc_curve(y_test, grid_gbdt.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, linestyle = 'solid', color = 'red')
plt.stackplot(fpr, tpr, color = 'steelblue')
plt.plot([0,1],[0,1], linestyle = 'dashed', color = 'black')
plt.text(0.6,0.4,'AUC=%.3f' % metrics.auc(fpr,tpr), fontdict = dict(size = 18))
plt.show()

