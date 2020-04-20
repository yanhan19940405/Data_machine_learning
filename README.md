# Data_machine_learning

##注意##
由于github某些组件最近被墙，导致图片无法正常加载，请按照如下方式处理：

1、更改hosts文件，添加如下信息，刷新网页即可解决此问题：

>> ```
>> 52.74.223.119 github.com
>> 192.30.253.119 gist.github.com
>> 54.169.195.247 api.github.com
>> 185.199.111.153 assets-cdn.github.com
>> 151.101.76.133 raw.githubusercontent.com
>> 151.101.108.133 user-images.githubusercontent.com
>> 151.101.76.133 gist.githubusercontent.com
>> 151.101.76.133 cloud.githubusercontent.com
>> 151.101.76.133 camo.githubusercontent.com
>> 52.74.223.119 github.com
>> 192.30.253.119 gist.github.com
>> 54.169.195.247 api.github.com
>> 185.199.111.153 assets-cdn.github.com
>> 151.101.76.133 raw.githubusercontent.com
>> 151.101.108.133 user-images.githubusercontent.com
>> 151.101.76.133 gist.githubusercontent.com
>> 151.101.76.133 cloud.githubusercontent.com
>> 151.101.76.133 camo.githubusercontent.com
>> ```

2、可以将源码图片文件夹中找到相应图片。

2019年CCF智能信用评分大赛个人源码库。包含XGboost模型调参，特征筛选，训练等方案。同时包含stacking模型融合方案

经过xgboost进行特征筛选，特征图如下：

![图1 feature select](https://github.com/yanhan19940405/Data_machine_learning/blob/master/image/Figure_3.png)

后续建模均使用此方式提取得特征建模，贡献度阈值设置为0.01

除此外，stacking_class.py是封装stacking 类代码，功能从特征提取到第一，第二层模型训练，到预测结束。

[简单跑分结果和详细解读请参阅](https://yanhan19940405.github.io/2019/03/12/%E6%A8%A1%E5%9E%8B%E8%9E%8D%E5%90%88%E6%96%B9%E6%A1%88Stacking%E5%8E%9F%E7%90%86%E4%B8%8E%E5%AE%9E%E7%8E%B0/)
