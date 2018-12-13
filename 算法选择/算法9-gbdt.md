## 算法原理-句话说明

梯度提升决策树-GBDT(Gradient Boosting Decision Tree)，通过对多颗决策树进行集成，从而进行预测的算法。

## 算法原理-文档

梯度提升决策树-GBDT(Gradient Boosting Decision Tree)，通过对多颗决策树进行集成，从而进行预测的算法。
在构造多颗决策树的时候，GBDT采用boosting的方法，通过对多颗决策树集成，可以获得比单个决策树更好的泛化能力/鲁棒性。




## 算法原理-参数
- n_estimators: 弱学习器(即回归树)的数量.
- max_depth：每个树的深度
- max_leaf_nodes：每个数的叶子节点数目。
- learning_rate：学习速度。这个参数取值在 (0,1]  之间，通过缩减步长来控制过拟合。




## 场景可视化-交互式

![](../_image/检测效果图.png)

