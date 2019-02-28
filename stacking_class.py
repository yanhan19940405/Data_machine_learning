#--coding=utf-8--
# coding=utf8
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
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets.samples_generator import make_blobs
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
'''创建训练的数据集'''

from sklearn.model_selection import KFold

'''5折stacking'''
class stacking_model:
    def __init__(self):
        print(1)
    def first_level_model(self,nfolds,train_data,target_data,test_data,clfs):#n_folds:K折次数，train_data:训练集,target_data:目标特征值列表，test_data:测试数据集，clfs：stacking基模型列表
        n_folds = 5
        train_set = []
        test_set = []
        target_set = []
        new_target_value = []
        new_feature_value = []
        new_feature_all = []
        new_train_value = []
        test_data_set = []
        test_id = []
        skf = KFold(n_splits=nfolds, shuffle=False).split(train_data)
        print(train_data[3].shape)
        print(train_data[3])
        for train, test in skf:
            for j, clf in enumerate(clfs):
                train_set = list(map(lambda a: list(train_data[a, :]), train))
                test_set = list(map(lambda a: list(train_data[a, :]), test))
                target_set = list(map(lambda a: target_data[a], train))
                train_set = pd.DataFrame(train_set)
                print(train_set.shape)
                print(len(target_set))
                print("model", clf)
                clf.fit(train_set, target_set)
                new_feature_target = clf.predict(test_set)
                new_feature_value.append((j, new_feature_target))
                test_id.append((j, test))
                test_data_set.append((j, clf.predict(test_data)))

        print("新的特征", test_id)
        print("新的特征序列", new_feature_value)
        print("测试数据集", test_data_set)
        return new_feature_value,test_data_set
    def second_model_train(self,new_feature_value,test_data_set):#返回基模型处理后的超特征矩阵，和测试数据集融合特征
        new_feature_0 = []
        new_feature_1 = []
        new_feature_2 = []
        new_feature_3 = []
        new_feature_4 = []
        new_feature_matrix = []
        for i in new_feature_value:
            if i[0] == 0:
                new_feature_0 = new_feature_0 + list(i[1])
            elif i[0] == 1:
                new_feature_1 = new_feature_1 + list(i[1])
            elif i[0] == 2:
                new_feature_2 = new_feature_2 + list(i[1])
            elif i[0] == 3:
                new_feature_3 = new_feature_3 + list(i[1])
            elif i[0] == 4:
                new_feature_4 = new_feature_4 + list(i[1])
        print(new_feature_1)
        new_feature_matrix.append(new_feature_0)
        new_feature_matrix.append(new_feature_1)
        new_feature_matrix.append(new_feature_2)
        new_feature_matrix.append(new_feature_3)
        new_feature_matrix.append(new_feature_4)
        new_feature_matrix = pd.DataFrame(new_feature_matrix)
        new_feature_matrix = new_feature_matrix.T
        print(new_feature_matrix)

        new_test_0 = []
        new_test_1 = []
        new_test_2 = []
        new_test_3 = []
        new_test_4 = []
        new_test_matrix = []
        for i in test_data_set:
            if i[0] == 0:
                new_test_0.append(list(i[1]))
            elif i[0] == 1:
                new_test_1.append(list(i[1]))
            elif i[0] == 2:
                new_test_2.append(list(i[1]))
            elif i[0] == 3:
                new_test_3.append(list(i[1]))
            elif i[0] == 4:
                new_test_4.append(list(i[1]))
        new_test_0 = (pd.DataFrame(new_test_0)).T
        new_test_1 = (pd.DataFrame(new_test_1)).T
        new_test_2 = (pd.DataFrame(new_test_2)).T
        new_test_3 = (pd.DataFrame(new_test_3)).T
        new_test_4 = (pd.DataFrame(new_test_4)).T
        j = 0
        B0 = []
        B1 = []
        B2 = []
        B3 = []
        B4 = []
        new_test = [new_test_0, new_test_1, new_test_2, new_test_3, new_test_4]
        for i in range(len(new_test)):
            for j in range(len(new_test[i])):
                if i == 0:
                    B0.append(np.mean(new_test[i].iloc[j, :]))
                elif i == 1:
                    B1.append(np.mean(new_test[i].iloc[j, :]))
                elif i == 2:
                    B2.append(np.mean(new_test[i].iloc[j, :]))
                elif i == 3:
                    B3.append(np.mean(new_test[i].iloc[j, :]))
                elif i == 4:
                    B4.append(np.mean(new_test[i].iloc[j, :]))
        new_test_matrix.append(B0)
        new_test_matrix.append(B1)
        new_test_matrix.append(B2)
        new_test_matrix.append(B3)
        new_test_matrix.append(B4)
        new_test_matrix = pd.DataFrame(new_test_matrix)
        new_test_matrix = new_test_matrix.T
        print(new_test_matrix)

        return new_feature_matrix,new_test_matrix
    def second_model_predict(self,new_feature_matrix,new_test_matrix,id):
        other_params = {'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 3, 'min_child_weight': 5, 'seed': 0,
                        'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'reg_alpha': 0, 'reg_lambda': 1,
                        'eval_metric': "mae", 'verbose': 1}
        clf_2 = XGBRegressor(**other_params)
        clf_2.fit(new_feature_matrix, target_data)
        test_pre = clf_2.predict(new_test_matrix)
        test_pre = [int(a) for a in test_pre]
        print(len(test_pre))
        dataframe = pd.DataFrame({'id': id, 'score': test_pre})
        dataframe.to_csv("result_ensemble_stack.csv", index=False, sep=',')
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    data = pd.read_csv('train_dataset.csv', encoding='utf-8')
    test = pd.read_csv('test_dataset.csv', encoding='utf-8')
    data = data
    data = data.fillna(0)
    data_hot = data.drop(columns=["用户编码", "信用分", "是否大学生客户", "用户实名制是否通过核实", "当月是否到过福州山姆会员店", "用户最近一次缴费距今时长（月）",
                                  "当月是否逛过福州仓山万达", "是否黑名单客户", "当月物流快递类应用使用次数",
                                  "是否经常逛商场的人", "是否4G不健康客户", "当月火车类应用使用次数", "当月是否看电影", "当月是否体育场馆消费"])
    data_hot = pd.DataFrame(data_hot, dtype=float)
    train_columns = data_hot.columns.values
    test_data = test.drop(columns=["用户编码", "是否大学生客户", "用户实名制是否通过核实", "当月是否到过福州山姆会员店", "用户最近一次缴费距今时长（月）",
                                   "当月是否逛过福州仓山万达", "是否黑名单客户", "当月物流快递类应用使用次数",
                                   "是否经常逛商场的人", "是否4G不健康客户", "当月火车类应用使用次数", "当月是否看电影", "当月是否体育场馆消费"])
    test_data = pd.DataFrame(test_data, dtype=float)
    target_data = data["信用分"]
    train_data = preprocessing.scale(data_hot)
    test_data = preprocessing.scale(test_data)
    ran_params = {'criterion': 'mae', 'max_depth': 3, 'n_estimators': 200, 'verbose': 1}
    logistic_params = {'penalty': 'l2', 'solver': 'lbfgs', 'verbose': 1}
    '''模型融合中使用到的各个单模型'''
    clfs = [LinearRegression(),
            ExtraTreesRegressor(**ran_params, bootstrap=True),
            RandomForestRegressor(**ran_params),
            GradientBoostingRegressor(**ran_params),
            LogisticRegression(**logistic_params)]
    id = test["用户编码"]
    new_feature_value, test_data_set=stacking_model().first_level_model(nfolds=5, train_data=train_data,target_data=target_data,test_data=test_data,clfs=clfs)
    new_feature_matrix, new_test_matrix=stacking_model().second_model_train(new_feature_value=new_feature_value,test_data_set=test_data_set)
    stacking_model().second_model_predict(new_feature_matrix=new_feature_matrix,new_test_matrix=new_test_matrix,id=id)