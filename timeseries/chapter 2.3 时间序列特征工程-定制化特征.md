[TOC]

# 引言

2.3介绍了常用的时序特征分类，但是在实际应用中，直接套用2.3的特征可能起不到很好的效果，这个时候需要针对业务问题设计定制化的特征。接下来会以运维场景常见的三个场景进行说明

- 时间序列异常检测
- 特殊模式识别
- 曲线分类

# 时序异常检测

异常检测是数据挖掘中一个重要方面，被用来发现小的 模式(相对于聚类)，即数据集中间显著不同于其它数据的对 象，异常检测应用在电信和信用卡欺骗、贷款审批、气象预报、金融领域和客户分类等领域中。
Hawkins[^3]给出了异常的本质性的定义:异常是在数据 集中与众不同的数据，使人怀疑这些数据并非随机偏差，而 是产生于完全不同的机制（regime） 。后来研究者们根据对异常存在的 不同假设，发展了很多异常检测算法，大体可以分为基于统 计的算法、基于深度的算法、基于距离的算法、基于密度的算法，以及面向高维数据的算法等。
总结一下，异常检测本质是从历史数据学到正常模式，并将检测数据跟正常模式进行比较，进而判断得出是否异常。基于该原理，我们可以设计对比类特征:

- 简单对比特征
- 统计量 \* 对比特征
- 窗口参数\* 统计量 \* 对比特征

##  简单对比特征

### 1.对比类（同环比）

对比类特征包含两个组成部分：参与比较的对象和距离函数。同环比特征是最简单通用的特征。

在传统的监控系统中经常对同环比设置阈值，如配置告警规则“同比昨天超过10%就告警”，我们的特征设计也可以参考这种监控策略。

- 差分

参与比较的对象：当前时刻的时序值和上个时刻的时序值，距离函数是差。

$$diff_t = y_t - y_{t-1} $$

- 环比

参与比较的对象：当前时刻的时序值和上个时刻的时序值，距离函数是比值。

$$rate_t=y_t/y_{t-1}$$

- 环比变化率

 参与比较的对象：当前时刻的时序值和上个时刻的时序值，距离函数是增长率。
  $$rate_t =( y_t - y_{t-1})/y_{t-1}$$

### 2.对比dist * 统计量s

当在波动率的维度去检测异常时，可以设计波动率差分特征和波动率环比特征：

波动率差分vol_diff:

${vol}_t = \sqrt{\frac{1}{w} \sum\limits_{i = 1}^w (y_{t-i+1} - \bar y_t)^2}​$

$f_t= {vol}_t - vol_{t-1} $

波动率环比vol_rate:

${vol}_t = \sqrt{\frac{1}{w} \sum\limits_{i = 1}^w (y_{t-i+1} - \bar y_t)^2}​$

$f_t= {vol}_t / vol_{t-1} $



### 3.对比dist* 统计量 s*窗口长度w



#### 3.1 比较长期均线和短期均线

设计长期窗口均值${\bar y}_{t, w_1}$和短期窗口均值${\bar y}_{t, w_2} $的差:

$${\bar y}_{t, w_1} = {\frac{1}{w_1} \sum\limits_{i = 1}^{w_1} y_{t-i+1} }$$

$${\bar y}_{t, w_2} = {\frac{1}{w_2} \sum\limits_{i = 1}^{w_2} y_{t-i+1} }$$

$$f_t= {\bar y}_{t, w_1}-{\bar y}_{t, w_2} $$
其中$w_1$和$w_2$是窗口长度, 并且$$w_1  > w_2$$。

如果$$f_t>0​$$，说明呈上升趋势
如果$$f_t<0​$$，说明呈3下降趋势


#### 3.2 历史均值差分 
历史均值偏移 (history_detector):计算当前窗口最后一个点跟历史窗口均值的差
$$| x_i - mean(X_{historywindow}))| ​$$

#### 3.3 历史斜率差分

历史斜率偏移(history_trend_detector)
计算当前窗口的斜率跟历史窗口斜率的差值
$$slope(X_{currenwindow}) - slope(X_{historywindow}))​$$

#### 3.4 历史相似性

历史相似性（correlation_distance）：计算当前窗口的跟历史窗口的相关系数
$$correlation(X_{currenwindow},X_{historywindow})) = \frac{\sum\limits_{t=1}^{n-1} (X_t- \bar X )(Y_{t}-\bar Y) }{\sqrt{\sum\limits_{t=1}^{n} (X_t- \bar X_t )^2}\sqrt{\sum\limits_{t=1}^{n} (y_t- \bar y_t )^2}}$$

## 基于模型的异常特征

###  控制图算法
控制图算法是一中质量控制领域的检测算法，可用于时序数据异常检测。对不光滑/信噪比很高的曲线，控制图算法能够适应信号中噪音成分，相比简单同环比检测策略，带来的误告更少。控制图算法中常用的算法有移动平均控制图，指数加权移动平均控制图，cum_sum控制图。这里主要介绍两种控制图算法:

- 1.移动平均控制图
- 2.指数加权平均控制图

#### 1.移动平均控制图
moving_average_detector|基于滑动平均值的异常得分|使用ma作为预测值，mstd作为置信区间，计算最后一个点的z_score，作为异常程度|$\frac{x_i - mean(X_{ma})}{std(X_{ma})} $|

#### 2.指数加权平均控制图
EWMA_detector|基于指数加权平均的异常得分|使用ewma作为预测值，ewmstd作为置信区间，计算异常得分，计算最后一个点的z_score，作为异常程度|$\frac{x_i - mean(X_{EWMA})}{std(X_{EWMA})} $|


### 拟合类特征

#### 1.基于最小二乘的异常分值

基于最小二乘的异常分值(least_squares_detector)：使用直线拟合该区间，并计算最近一个时刻真实值和拟合值之间的差值，作为异常得分。
$$\hat x_i - x_i = least\_square\_predict(x_i) - x_i = (X^T X)^{-1}XY * X_i - x_i$$


#### 2.基于高斯分布的异常得分

基于高斯分布的异常得分(Gaussian_detector): 假设窗口数据服从高斯分布，计算最后一点在在分布上出现的概率。


#### 3.基于绝对离差中位数的异常得分

基于绝对离差中位数的异常得分(mad_detector):检测最后一个点偏离整个窗口的程度,

$$| x_i - mad(X_{curren\_window})) |$$



#  特殊模式识别

特殊模式识别指的是曲线中具有某种规律性的模式需要被识别出来，比如定时任务或者营销活动造成的在线曲线周期性的下降或上升，常用到的特征有：

## 1.是否出现陡增(spike)/陡降
judge_crash_habit
是否出现突变(judge_crash_habit)：是否出现陡降/陡增。具体的计算逻辑：最近时间窗口波动超过20%内或者稍远窗口波动超过40%,

 $$vol_{near} > 20% $$or$$ vol_{far} > 40% $$


##  2.两个序列的最短距离

两个序列的最短距离(least_distance_index_after_shift),考虑时间偏置后的两个序列的最短距离:
$$\min\limits_{off\_set}distance(X_{current\_window}, shift(X_{history\_window}, off\_set)​$$


## 3.波峰/波谷

从人眼看很容易指出波峰波谷在哪里，那怎么把人的经验通过计算机语言表达出来呢？

首先引入一个概念-波峰

波峰（peaks）：Use `findpeaks` to find values and locations of local maxima in a set of data.

波谷（valley）：Use `findpeaks` to find values and locations of local maxima in a set of data.



如何检测出波峰？需要用到一个数学工具：光滑曲线上在某个闭区间[a, b], 如果a的导数大于0，b的导数小于0，则必定在[a, b]中间有一个局部最大值；

如果a的导数小于0，b的导数大于0，则必定在[a, b]中间有一个局部最小值。



有了这个理论指导，我们需要一步一步来实现异常特征构造：

1. 怎么表达数学中“导数”的概念？

   一阶差分序列

     $$ \Delta X_t= x_{t} - x_{t-1}$$

2. 为了判断t时刻的点是否为peak：

   构造1中的一阶差分序列$$\Delta X_t$$，再计算一次差分：

     $$ \Delta^2 X_{t+1} = \Delta X_{t+1} - \Delta X_t =x_{t+1} - 2x_{t} +  x_{t}$$

   当 $$\Delta^2 X_{t+1} > 0$$ 时，表明t时刻为波峰

   当$$\Delta^2 X_{t+1} < 0$$ 时，表明t时刻为波谷


需要注意的是，在对流式数据检测异常时，在t时刻获取到数据$$x_t$$后，只能判断t-1时刻是不是波峰/波谷，存在1个时刻的延迟。


```python
doublediff = np.diff(np.sign(np.diff(data)))
peak_locations = np.where(doublediff == -2)[0] + 1

doublediff2 = np.diff(np.sign(np.diff(-1*data)))
trough_locations = np.where(doublediff2 == -2)[0] + 1
```

![image-20190115000606151](/Users/stellazhao/statistics_studyplace/EasyML_BOOK/_image/image-20190115000606151.png)

```matlab
load(fullfile(matlabroot,'examples','signal','spots_num.mat'))

[pks,locs] = findpeaks(avSpots);

plot(year,avSpots,year(locs),pks,'or')
xlabel('Year')
ylabel('Number')
axis tight
```

![image-20190115002205289](/Users/stellazhao/statistics_studyplace/EasyML_BOOK/_image/image-20190115002205289.png)

## 4.是否为拐点

是否为交叉点(CrossingPoint_detector)：

计算是否是局部拐点，比较最近一段曲线 和 倒数第二段曲线的方向是否一致|$is\_same\_direction(X_{near\_window}, X_{far\_window})$|


## 5. 窗口最近一个点的增长率
窗口最近一个点的增长率(ChangePoint_detector)
ChangePoint_detector|突变得分|计算窗口最近一个点的增长率/斜率|$slope(x_i)$|
diff2|二阶差分|二阶差分|$x_i - 2 * x_{i-1} + x_{i-2}$|



# 曲线关键特性

时间序列曲线分类场景中，需要从周期性，毛刺程度，光滑程度等统计性质去考虑不同曲线的差异性，常用的特征有以下类型。

## 周期性

特征名|特征中文名|特征描述|计算公式|
:---------------------|------------------|-------------|---------|---------
similarity_detween_days| 天跟天的相似相似性|天跟天的相似相似性的平均值 |$\frac{1}{2D}\sum\limits_{i!=j,i=1,j=1}^D similarity(DAY_i, DAY_j)$|
period_1day|天周期性|lombscargle统计量|lombscargle_score(1440)|
period_1week|周周期性|lombscargle统计量|lombscargle_score(1440 * 7)|
acf_value_1d|滞后时间为1天的时序自相关系数|lag=1440的自相关系数,默认时序的频率为1分钟|$acf_{1440}$|
acf_value_7d|滞后时间为7天的时序自相关系数|lag=1440 * 7 的自相关系数, 默认时序的频率为1分钟|$acf_{1440 * 7}$|




## 光滑

特征名|特征中文名|特征描述|计算公式|
:---------------------|------------------|-------------|---------|---------|
decomp_ratio|时序分解的信噪比|使用tsd分解后，残差能量比总体能量|$energe(X_{残差})/energe(X)$|
ratio_crossing_point|拐点在窗口中占比|拐点在窗口中占比|$\frac {拐点个数}{X的长度}$|
ratio_change_point|突变点在窗口中占比|突变点在窗口中占比|$\frac {突变点个数}{X的长度}$|
volicity|一阶差分平方和|一阶差分平方和| $\|X_{diff}\|_2^2$|
vol|一阶差分标准差|一阶差分标准差|$std(X_{diff})$|



怎样去评估

![image-20190120110327181](/Users/stellazhao/tencent_workplace/gitlab/dataming/algorithm_doc/process-model/_image/ac.png)

![image-20190120110142530](/Users/stellazhao/tencent_workplace/gitlab/dataming/algorithm_doc/process-model/_image/diff_locs.png)


# 参考资料


[^3]: Hawkins D.Identification of Outliers.Chapman and Hall,London,1980

https://ww2.mathworks.cn/help/signal/getting-started-with-signal-processing-toolbox.html
https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/?utm_campaign=shareaholic&utm_medium=twitter&utm_source=socialnetwork