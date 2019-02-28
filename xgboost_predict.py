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
id=test["用户编码"]
test_data=test.drop(columns=["用户编码","是否大学生客户","用户实名制是否通过核实","当月是否到过福州山姆会员店","用户最近一次缴费距今时长（月）",
                            "当月是否逛过福州仓山万达","是否黑名单客户","当月物流快递类应用使用次数",
                            "是否经常逛商场的人","是否4G不健康客户","当月火车类应用使用次数","当月是否看电影","当月是否体育场馆消费"])
test_data=pd.DataFrame(test_data,dtype=float)
target_data=data["信用分"]
print(data_hot.shape)
print(target_data)
train_data = preprocessing.scale(data_hot)
test_data=preprocessing.scale(test_data)
print(train_data.shape)
X_train, X_test, y_train, y_test = train_test_split(train_data,target_data, train_size=0.8, test_size=0.2)
# # model=RandomForestClassifier(n_estimators=10, max_depth=5,random_state=24,criterion="gini")
cv_params = {'n_estimators': [1000,1500,2000,3000,3500,4000,4500,5000,5500,6000],'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
other_params = {'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 5,'min_child_weight': 3,
              'gamma': 0.1,'eval_metric':"mae"}

model = XGBRegressor(**other_params)
model.fit(X_train, y_train,verbose=True)
# model=joblib.load( "train_model.m")
pre=model.predict(X_test)
a=mean_absolute_error(y_test, pre)
score=(1/(a+1))
print(score)
test_pre=model.predict(test_data)
test_pre=[int(a)for a in test_pre]
dataframe = pd.DataFrame({'id': id, 'score': test_pre})
dataframe.to_csv("result_xgb_3.csv", index=False, sep=',')

print(1)

