# 时序异常检测

## 引言

时序异常检测就是对时间序列数据监测出异常的模式。

## 技术难点

很难用一些策略去构建一个通用的异常检测服务，因为监控指标各异（正常模式各异），异常各异（异常类型多种）。其中，比较难识别的几个曲线和异常如下：

1.历史数据有中断缺失
指标：在线数据由于数据质量有中断缺失
正常模式： 历史数据有缺失
检测策略：先使用正常数据填充再检测。

![image-20190114173717116](/Users/stellazhao/statistics_studyplace/EasyML_BOOK/_image/image-20190114173717116.png)

2.整体趋势变化：趋势漂移是正常模式。
指标： 游戏收入
正常模式：游戏收入在开学季趋势漂移。
检测策略：学习并且剔除这个趋势漂移。



![image-20190114173747851](/Users/stellazhao/statistics_studyplace/EasyML_BOOK/_image/image-20190114173747851.png)



3.指标： 定时任务日志数
正常模式：周期性有数据
检测策略：检测规律行为数据缺失

![image-20190114174212541](/Users/stellazhao/statistics_studyplace/EasyML_BOOK/_image/image-20190114174212541.png)



4.合理范围的突变异常。
指标：登录在线等周期性曲线
正常模式：趋势呈周期性
检测策略：检测突变点。

![image-20190114181500368](/Users/stellazhao/statistics_studyplace/EasyML_BOOK/_image/image-20190114181500368.png)



5.无规律指标识别。
指标：毫无规律指标
正常模式：在一定统计范围内波动
异常检测策略：相对历史分布的极端异常值。

![image-20190114181832710](/Users/stellazhao/statistics_studyplace/EasyML_BOOK/_image/image-20190114181832710.png)



6.指标：跑批数据
正常模式：周期性毛刺点，但是周期间隔不明显，可能会有偏移。
检测策略：使用高斯核函数拟合分布极值？

1. 使用dtw去衡量两个窗口的距离，兼容偏差。
2. 在历史数据中使用滚动窗口的方法，找到一个最近的窗口。

![image-20190114182211015](/Users/stellazhao/Library/Application Support/typora-user-images/image-20190114182211015.png)



7. 周期性明显数据
指标：银行的业务数据
正常模式：周期性跌零，周期性有数据

![image-20190114182435743](/Users/stellazhao/statistics_studyplace/EasyML_BOOK/_image/image-20190114182435743.png)





如何获取曲线的关键特征：

1. 周期性： autocorrelation
2. 周期offset：高斯核函数拟合分布举止
3. 趋势判断：指数滑动平均
4. 分析数据极值: 假设检验。



![image-20190114190407015](/Users/stellazhao/statistics_studyplace/EasyML_BOOK/_image/image-20190114190407015.png)

