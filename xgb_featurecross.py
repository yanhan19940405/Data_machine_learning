import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # for plotting
from scipy.stats import mode
import warnings   # to ignore warnings
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score#R square
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import re
import seaborn as sns
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
data = pd.read_csv('train_dataset.csv',encoding='utf-8')
test=pd.read_csv('test_dataset.csv',encoding='utf-8')
data=data
data=data.fillna(0)
data_hot=data.drop(columns=["用户编码","信用分","是否大学生客户","用户实名制是否通过核实","当月是否到过福州山姆会员店","用户最近一次缴费距今时长（月）",
                            "当月是否逛过福州仓山万达","是否黑名单客户","当月物流快递类应用使用次数",
                            "是否经常逛商场的人","是否4G不健康客户","当月火车类应用使用次数","当月是否看电影","当月是否体育场馆消费"])
data_hot=pd.DataFrame(data_hot,dtype=float)

test_data=test.drop(columns=["用户编码"])
test_data=pd.DataFrame(test_data,dtype=float)
target_data=data["信用分"]
print(data_hot.shape)
print(target_data)
train_data = preprocessing.scale(data_hot)
test_data=preprocessing.scale(test_data)
print(train_data.shape)
X_train, X_test, y_train, y_test = train_test_split(train_data,target_data, train_size=0.8, test_size=0.2)
# # model=RandomForestClassifier(n_estimators=10, max_depth=5,random_state=24,criterion="gini")
cv_params = {'learning_rate':[0.01,0.02,0.05,0.1,0.2,0.3,0.4],'gama':[0.1,0.2,0.3,0.4],
             'n_estimators': [100,200,300,400,500,600,700,800,900,1000,1500,2000],'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
other_params = {'learning_rate': 0.1, 'n_estimators': 800, 'max_depth': 3,'min_child_weight': 5,'seed': 0,
               'gamma': 0.1,'reg_alpha': 0,'reg_lambda': 1,'eval_metric':"mae"}

model = XGBRegressor(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params,scoring='neg_median_absolute_error', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, y_train)
evalute_result = optimized_GBM.cv_results_['mean_test_score']
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

# model.fit(X_train,y_train)
# joblib.dump(model, "train_model.m")
# pre=model.predict(X_test)
# from sklearn.metrics import mean_absolute_error
# mre= mean_absolute_error(y_test, pre)
# score=1/(1+mre)
# print("score",score)
#model=joblib.load( "train_model.m")
# feature_importance = model.feature_importances_
# result1 = np.argsort(feature_importance)
# result=result1[0:10]
# plt.figure(figsize=(64, 64))
# plt.barh(range(len(result1)), feature_importance[result1], align='center')
# plt.yticks(range(len(result1)), train_columns[result1])
# plt.xlabel("变量权重图")
# plt.title('Feature importances')
# plt.draw()
# plt.show()
# print("变量权重如下",feature_importance[result1])
# print("的关键变量如下所示：",train_columns[result1])
print(1)

