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
data_hot=data.drop(columns=["用户编码","信用分"])
data_hot=pd.DataFrame(data_hot,dtype=float)
train_columns=data_hot.columns.values
test_data=test.drop(columns=["用户编码"])
test_data=pd.DataFrame(test_data,dtype=float)
target_data=data["信用分"]
train_data = preprocessing.scale(data_hot)
test_data=preprocessing.scale(test_data)

print(test_data)
print(train_data)
X_train, X_test, y_train, y_test = train_test_split(train_data,target_data, train_size=0.8, test_size=0.2)
other_params = {'learning_rate': 0.1, 'n_estimators': 5000, 'max_depth': 3,'min_child_weight': 5,'seed': 0,
               'subsample': 0.8, 'colsample_bytree': 0.8,'gamma': 0.1,'reg_alpha': 0,'reg_lambda': 1,'eval_metric':"mae"}

model = XGBRegressor(**other_params)
model.fit(X_train,y_train)
feature_importance = model.feature_importances_
result1 = np.argsort(feature_importance)
result=result1[0:10]
plt.figure(figsize=(64, 64))
plt.barh(range(len(result1)), feature_importance[result1], align='center')
plt.yticks(range(len(result1)), train_columns[result1])
plt.xlabel("变量权重图")
plt.title('Feature importances')
plt.draw()
plt.show()
print("变量权重如下",feature_importance[result1])
print("关键变量如下所示：",train_columns[result1])
print(1)

