# Data_machine_learning
2019年CCF智能信用评分大赛个人源码库。包含XGboost模型调参，特征筛选，训练等方案。同时包含stacking模型融合方案

经过xgboost进行特征筛选，特征图如下：

![图1 feature select](https://github.com/yanhan19940405/Data_machine_learning/blob/master/image/Figure_3.png)

后续建模均使用此方式提取得特征建模，贡献度阈值设置为0.01

除此外，stacking_class.py是封装stacking 类代码，功能从特征提取到第一，第二层模型训练，到预测结束。
