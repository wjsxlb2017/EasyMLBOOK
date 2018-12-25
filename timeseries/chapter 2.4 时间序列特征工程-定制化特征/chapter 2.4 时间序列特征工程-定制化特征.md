2.3介绍了常用的时序特征分类，但是在实际应用中，直接套用2.3的特征可能起不到很好的效果，这个时候需要针对业务问题设计定制化的特征。接下来会以运维场景常见的三个场景进行说明

- 曲线分类
- 时间序列异常检测
- 特殊模式识别

## 1.异常检测

异常检测是数据挖掘中一个重要方面，被用来发现小的 模式(相对于聚类)，即数据集中间显著不同于其它数据的对 象，异常检测应用在电信和信用卡欺骗、贷款审批、气象预报、金融领域和客户分类等领域中。
Hawkins[^3]给出了异常的本质性的定义:异常是在数据 集中与众不同的数据，使人怀疑这些数据并非随机偏差，而 是产生于完全不同的机制（regime） 。后来研究者们根据对异常存在的 不同假设，发展了很多异常检测算法，大体可以分为基于统 计的算法、基于深度的算法、基于距离的算法、基于密度的算法，以及面向高维数据的算法等。
总结一下，异常检测本质是从历史数据学到正常模式，并将检测数据跟正常模式进行比较，进而判断得出是否异常。基于该原理，我们可以设计对比类特征:

- 简单对比特征
- 统计量 \* 对比特征
- 窗口参数\* 统计量 \* 对比特征

### 特征设计方法举例
####  简单对比特征

差分

$diff_t = y_t - y_{t-1} $

环比

$rate_t = y_t / y_{t-1}$ 

增长率

$rate_t =( y_t - y_{t-1}) / y_{t-1}$ 

#### 时序属性 * 对比特征

当在波动率的维度去检测异常时，可以设计波动率差分特征和波动率环比特征：

波动率差分vol_diff:

${vol}_t = \sqrt{\frac{1}{w} \sum\limits_{i = 1}^w (y_{t-i+1} - \bar y_t)^2}$

$f_t= {vol}_t - vol_{t-1} $

波动率环比vol_rate:

${vol}_t = \sqrt{\frac{1}{w} \sum\limits_{i = 1}^w (y_{t-i+1} - \bar y_t)^2}$

$f_t= {vol}_t / vol_{t-1} $



#### 多窗口长度 * 统计量 * 对比特征
对于毛刺曲线，如何找突变点？

设计长期窗口均值${\bar y}_{t, w_1}$和短期窗口均值${\bar y}_{t, w_2} $的差分(diff):

$${\bar y}_{t, w_1} = {\frac{1}{w_1} \sum\limits_{i = 1}^{w_1} y_{t-i+1} }$$

$${\bar y}_{t, w_2} = {\frac{1}{w_2} \sum\limits_{i = 1}^{w_2} y_{t-i+1} }$$

$$f_t= {\bar y}_{t, w_1}-{\bar y}_{t, w_2} $$
其中$w_1$和$w_2$是窗口长度。

### 异常特征总结

$$\left| x_i - mad(X_{curren\_window})) \right|$$


特征名|特征中文名|特征描述|计算公式|
----------------------|------------------|-------------|---------|
mad_detector | 基于绝对离差中位数的异常得分|检测最后一个点偏离整个窗口的程度, 类似z_score|$\left| x_i - mad(X_{curren\_window})) \right|$|
history_detector|历史均值差分 | 计算当前窗口最后一个点跟历史窗口均值的差|$\left| x_i - mean(X_{history\_window}))\right |$|
history_trend_detector| 历史斜率差分|计算当前窗口的斜率跟历史窗口斜率的差值|$slope(X_{curren\_window}) - slope(X_{history\_window}))$|
correlation_distance|历史相似性|计算当前窗口的跟历史窗口的相关系数|$correlation(X_{curren\_window},X_{history\_window})) = \frac{\sum\limits_{t=1}^{n-1} (X_t- \bar X )(Y_{t}-\bar Y) }{\sqrt{\sum\limits_{t=1}^{n} (X_t- \bar X_t )^2}\sqrt{\sum\limits_{t=1}^{n} (y_t- \bar y_t )^2}}$|
moving_average_detector|基于滑动平均值的异常得分|使用ma作为预测值，mstd作为置信区间，计算最后一个点的z_score，作为异常程度|$\frac{x_i - mean(X_{ma})}{std(X_{ma})} $|
EWMA_detector|基于指数加权平均的异常得分|使用ewma作为预测值，ewmstd作为置信区间，计算异常得分，计算最后一个点的z_score，作为异常程度|$\frac{x_i - mean(X_{EWMA})}{std(X_{EWMA})} $|
CrossingPoint_detector|是否为交叉点|计算是否是局部拐点，比较最近一段曲线 和 倒数第二段曲线的方向是否一致|$is\_same\_direction(X_{near\_window}, X_{far\_window})$|
least_squares_detector|基于最小二乘拟合的一场得分|使用直线拟合该区间，并计算最近一个时刻真实值和拟合值之间的差值，作为异常得分|$\hat x_i - x_i = least\_square\_predict(x_i) - x_i = (X^T X)^{-1}XY * X_i - x_i$|
ChangePoint_detector|突变得分|计算窗口最近一个点的增长率/斜率|$slope(x_i)$|
diff2|二阶差分|二阶差分|$x_i - 2 * x_{i-1} + x_{i-2}$|

## 曲线分类

时间序列曲线分类场景中，需要从周期性，毛刺程度，光滑程度等层面去考虑不同曲线的差异性，常用的特征有一下类型。

### 曲线类型特征
特征名|特征中文名|特征描述|计算公式|
----------------------|------------------|-------------|---------|
similarity_detween_days| 天跟天的相似相似性|天跟天的相似相似性的平均值 |$\frac{1}{2D}\sum\limits_{i!=j,i=1,j=1}^D similaryty(DAY_i, DAY_j)$|
period_1day|lombscargle统计量|lombscargle统计量|lombscargle_score(1440)
period_1week|周周期性|lombscargle统计量|lombscargle_score(1440 * 7)||
acf_value_1d|滞后时间为1天的时序自相关系数|lag=1440的自相关系数,默认时序的频率为1分钟|$acf_{1440}$|
acf_value_7d|滞后时间为7天的时序自相关系数|lag=1440 * 7 的自相关系数, 默认时序的频率为1分钟|$acf_{1440 * 7}$|
decomp_ratio|时序分解的信噪比|使用tsd分解后，残差能量比总体能量|$energe(X_{残差})/energe(X)$|
ratio_crossing_point|拐点在窗口中占比|拐点在窗口中占比|$\frac {拐点个数}{X的长度}$|
ratio_change_point|突变点在窗口中占比|突变点在窗口中占比|$\frac {突变点个数}{X的长度}$
volicity|一阶差分平方和|一阶差分平方和| $\|X_{diff}\|_2^2$|
vol|一阶差分标准差|一阶差分标准差|$std(X_{diff})$|

## 特殊模式识别
特殊模式识别指的是曲线中具有某种规律性的模式需要被识别出来，比如定时任务或者营销活动造成的在线曲线周期性的下降或上升，常用到的特征有：


### 特殊模式识别特征
特征名|特征中文名|特征描述|计算公式|
----------------------|------------------|-------------|---------|
judge_crash_habit| 是否出现突变|是否出现陡降/陡增 |最近波动超过20%内或者稍远波动超过40%, $vol_{near} > 20% or vol_{far} > 40%$|
least_distance_index_after_shift|两个序列的最短距离|考虑时间偏置后的两个序列的最短距离|$\min\limits_{off\_set}distance(X_{current\_window}, shift(X_{history\_window}, off\_set)$|





# 参考资料


[^3]: Hawkins D.Identification of Outliers.Chapman and Hall,London,1980