
[TOC]
# 基于 TensorFlow.Probability的结构时间序列建模



在这篇文章中，我们介绍了 tfp.sts，这是一个新的应用结构时间序列模型预测时间序列的  [TensorFlow Probability](https://www.tensorflow.org/probability/)。

## 概览

> "做出预测很难，尤其是对未来的预测。"— [Karl Kristian Steincke 卡尔 · 克里斯蒂安 · 斯坦克](https://quoteinvestigator.com/2013/10/20/no-predict/)

尽管对未来事件的预测必然是不确定的，但预测是对未来进行规划的关键部分。 网站所有者需要预测他们网站的访问量，以便提供足够的硬件资源，并预测未来的收入和成本。 企业需要预测未来对消费产品的需求，以保持足够的产品库存。 电力公司需要预测电力需求，明智地购买能源合同，并建设新的发电厂。

预测时间序列的方法也可以用来推断特征发射或其他干预对用户参与度量[1]的因果影响，从更容易获得的信息[2]中推断难以观察的数量的当前值，以及检测时间序列数据中的异常。

## 结构时间序列

结构时间序列(STS)模型[3]是一系列时间序列的概率模型，它包括并概括了许多标准的时间序列建模思想，包括:

- autoregressive processes, 自回归过程,
- moving averages, 移动平均线
- local linear trends, 局部线性趋势
- seasonality, and 季节性，以及
- regression and variable selection on external covariates (other time series potentially related to the series of interest). 回归和外部协变量的变量选择(其他时间序列可能与感兴趣的序列相关)

Sts 模型将观测到的时间序列表示为简单成分之和:

每个组成部分都是由一个特定的结构假设支配的时间序列。 例如，一个组成部分可能编码成季节效应(seasonal。例如，星期几效应) ，另一个编码成局部线性趋势，还有一个编码成某些协变量时间序列的线性依赖项。

通过允许建模者对产生数据的过程进行编码假设，结构时间序列通常可以从相对较少的数据(例如，只有一个数十个点的单一输入序列)产生合理的预测。 模型的假设是可以解释的，我们可以通过将过去的数据和未来的预测可视化分解为结构成分来解释预测。 此外，结构时间序列模型使用概率公式，可以自然地处理缺失数据，并对不确定性进行量化。

## 使用TensorFlow Probability对Structural Time Series建模



Tensorflow 概率(TFP)现在内置支持使用结构时间序列模型进行拟合和预测。 这种支持包含使用变分推理(VI)和哈密顿蒙特卡罗(HMC)进行模型参数的贝叶斯推断，同时计算点预测和预测不确定性。

> 由于这些方法都是建立在 TensorFlow 上的，所以它们很自然地利用了矢量化的硬件(gpu 和 TPUs) ，能够有效地并行处理多个时间序列，并且能够与深层神经网络集成。

### 例子1: 二氧化碳浓度的预测

为了了解结sts的作用，我们来看看这个夏威夷莫纳罗亚火山天文台空间站提供的每月大气二氧化碳浓度记录



![img](https://cdn-images-1.medium.com/max/800/0*PQ09cqE6xNJZAf5G)

通过检查可以清楚地看出，这个系列既包含长期趋势（long-term trend ），也包含年度季节性变化（seasonal）。 我们可以用一个结构化的时间序列模型直接对这两个部分进行编码，只需要使用几行 TFP 代码:

```python

import tensorflow_probability as tfp
trend = tfp.sts.LocalLinearTrend(observed_time_series=co2_by_month)
seasonal = tfp.sts.Seasonal(
    num_seasons=12, observed_time_series=co2_by_month)
model = tfp.sts.Sum([trend, seasonal], observed_time_series=co2_by_month)
```

这里我们使用了一个局部线性趋势模型，它假设趋势是线性的，随着时间的推移，斜率在随机游走之后缓慢演化。 根据我们的建模假设，将模型与数据进行拟合得到一个概率预测:



![img](https://cdn-images-1.medium.com/max/800/0*Lgachj7sD9PBnfAR)

我们可以看到，随着时间的推移，预测的不确定性(阴影表示2个标准差内)增加，因为线性趋势模型对其斜率外推的信心变得越来越差。 平均预报结合了季节变化和现有趋势的线性外推，这似乎略微低估了大气 CO2的加速增长，但真实值仍然在95% 的预测区间内。

这个例子的完整[代码](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/jupyter_notebooks/Structural_Time_Series_Modeling_Case_Studies_Atmospheric_CO2_and_Electricity_Demand.ipynb) 可以在 Github 上找到。



### 例子2: 电力需求预测



接下来我们将考虑一个更复杂的例子: 预测澳大利亚维多利亚的电力需求。 这张图的顶线显示了2014年前六个星期的每小时记录(数据来自[4] ，可在 https://github.com/robjhyndman/fpp2-package 获得) :



![img](https://cdn-images-1.medium.com/max/800/0*MKIeUw0BDmjLhGX_)

在这里，我们可以访问外部信息来源: 温度，这与空气调节的电力需求有关。 记住，一月是澳大利亚的夏天！ 让我们把这些温度数据整合到一个 STS 模型中，这个模型可以通过线性回归来包含外部协变量:

请注意，我们还包括了多重季节性效应（ multiple seasonality effects）: hour-of-day，day-of-week effect，以及一个自回归的成分来对任何无法解释的残余效应进行建模。

 我们可能使用一个简单的随机游走，但是选成了一个自回归的组成部分，因为它的方差随着时间维持有界。



![img](/Users/stellazhao/EasyML_BOOK/_image/0*IlrfzF-A1tkCVGO8.png)

这个模型的预测并不完美ーー显然还有一变化没有被模型描述出来ーー但这并不疯狂，而且这些不确定性看起来也是合理的。 我们可以通过可视化分解成组件(注意每个组件图都有不同的 y 轴尺度)来更好地理解这个预测:



![image-20190513210842428](/Users/stellazhao/EasyML_BOOK/_image/image-20190513210842428.png)

我们看到，该模型已经相当合理地确定了一个大的小时效应和一个小得多的星期几效应(最低的需求似乎发生在星期六和星期日) ，以及一个相当大的温度效应，并且它产生了对这些效应的相对自信的预测。 绝大部分的预测不确定性来自于自回归过程，基于它对观测序列中未建模(残差)变化的估计。

建模者可能会使用这种分解来理解如何改进模型。 例如，他们可能会注意到，一些温度峰值似乎仍然与 AR 残差峰值重合，这表明额外的特征或数据转换可能有助于更好地捕捉温度效应。

这个例子的完整代码可以在 [Github](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/jupyter_notebooks/Structural_Time_Series_Modeling_Case_Studies_Atmospheric_CO2_and_Electricity_Demand.ipynb) 上找到。

## Tensorflow probability的STS 库

如上面的示例所示，TFP 中的 STS 模型是通过将模型组件添加到一起来构建的。 Sts 提供了如下建模组件:

- [Autoregressive 自回归](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/Autoregressive), [LocalLinearTrend 局部线性局势](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/LocalLinearTrend), [SemiLocalLinearTrend 半局部线性趋势](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/SemiLocalLinearTrend), and ，及[LocalLevel 局部水平](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/LocalLevel). For modeling time series with a level or slope that evolves according to a random walk or other process. . 用于建模具有水平（level）或斜率的时间序列，该时间序列根据随机漫步或其他过程进化
- [Seasonal 季节性的](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/Seasonal). 取决于季节因素的时间序列，如一天的小时，一周的天，或一年的月份
- [LinearRegression 线性回归](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/LinearRegression). 对于依赖于附加的时变协变量的时间序列。 回归分量也可用于编码假日或其他特定日期的效果


Sts 提供了用[变分推理](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/build_factored_variational_loss) 和[哈密顿蒙特卡罗方法](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/fit_with_hmc).拟合时间序列模型的方法。


在 [TFP](https://www.tensorflow.org/probability/). 主页上查看我们的代码、文档和进一步的示例。

Structural time series are being used for several important time series applications inside Google. We hope you will find them useful, as well. Please join the [tfprobability@tensorflow.org ]forum for the latest Tensorflow Probability announcements and other TFP discussions!

结构时间序列在谷歌内部被用于几个重要的时间序列应用。 我们希望你会发现它们也是有用的。 请加入 tfprobability@Tensorflow. org [论坛](https://groups.google.com/a/tensorflow.org/forum/#!forum/tfprobability)，了解最新的 Tensorflow 概率公告和其他 TFP 讨论！


## 参考资料

[1] Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., & Scott, S. L. (2015). Inferring causal impact using Bayesian structural time-series models. *The Annals of Applied Statistics*, *9*(1), 247–274.

[1] Brodersen，k. h. ，Gallusser，f. ，Koehler，j. ，Remy，n. ，& Scott，s. l. (2015). 利用贝叶斯结构时间序列模型推断因果影响。 应用统计年鉴，9(1) ，247-274。

[2] Choi, H., & Varian, H. (2012). Predicting the present with Google Trends. Economic Record, 88, 2–9.

 用谷歌趋势预测当前。 经济纪录，88,2-9。

[3] Harvey, A. C. (1989). *Forecasting, structural time series models and the Kalman filter*. Cambridge University Press.

[3]哈维，a. c. (1989)。 预测，结构时间序列模型和卡尔曼滤波。 剑桥大学出版社。

[4] Hyndman, R.J., & Athanasopoulos, G. (2018). Forecasting: principles and practice, 2nd edition, OTexts: Melbourne, Australia. OTexts.com/fpp2. Accessed on February 23, 2019.

[4] Hyndman，r.j. ，and Athanasopoulos，g. (2018). 预测: 原理与实践，第二版，otext: 澳大利亚墨尔本。 Otexts. com / fp2. 23,2019.

[5] Keeling, C. D., Piper, S. C., Bacastow, R. B., Wahlen, M., Whorf, T. P., Heimann, M., & Meijer, H. A. (2001). Exchanges of atmospheric CO2 and 13CO2 with the terrestrial biosphere and oceans from 1978 to 2000. I. Global aspects, SIO Reference Series, №01–06, Scripps Institution of Oceanography, San Diego.

[5] Keeling，C.d. ，Piper，S.c. ，Bacastow，R.b. ，Wahlen，m. ，Whorf，T.p. ，Heimann，m. ，& Meijer，h. a. (2001). 1978-2000年大气 CO2和13CO2与陆地生物圈和海洋的交换。 全球方面，SIO 参考系列，01-06，斯克里普斯海洋研究所，圣地亚哥。

