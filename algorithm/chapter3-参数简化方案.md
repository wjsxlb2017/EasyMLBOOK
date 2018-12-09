[TOC]

# 1. 简化方案汇总

算法名称 |训练阶段简化参数|应用阶段参数|
--|--|--|--
高斯分布检测器|-|标准差倍数|
滑动平均MA|滑动窗口长度|标准差倍数|
指数加权滑动平均|衰减因子|标准差倍数|
KNN|邻近点个数K|阈值|
DBSCAN|邻域的大小$\epsilon$ ;core point附近的点数min_samples|阈值|
LOF(local outlier factor)|邻近点个数K|LOF得分|
单分类SVM|nu|阈值alpha|
孤立森林|K|阈值a|


# 2.参考文献

[^1]: Outlier Detection Using Replicator Neural Networks， Simon Hawkins, Hongxing He, Graham Williams and Rohan Baxter


