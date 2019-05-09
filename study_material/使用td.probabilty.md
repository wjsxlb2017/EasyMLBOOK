

# 基于 TensorFlow.Probability的结构时间序列建模





[TensorFlow](https://medium.com/@tensorflow)

Mar 21 3月21日

*Posted by Dave Moore, Jacob Burnim, and the TFP Team*

作者: Dave Moore，Jacob Burnim，以及 TFP 团队

In this post, we introduce `tfp.sts`, a new library in [TensorFlow Probability](https://www.tensorflow.org/probability/)for forecasting time series using structural time series models [3].

在这篇文章中，我们介绍了 tfp.sts，这是一个新的应用结构时间序列模型预测时间序列的 TensorFlow 概率库[3]。

### Overview

### 概览

> “It is difficult to make predictions, especially about the future.” "很难做出预测，尤其是对未来的预测。"

> — [Karl Kristian Steincke 卡尔 · 克里斯蒂安 · 斯坦克](https://quoteinvestigator.com/2013/10/20/no-predict/)

Although predictions of future events are necessarily uncertain, forecasting is a critical part of planning for the future. Website owners need to forecast the number of visitors to their site in order to provision sufficient hardware resources, as well as predict future revenue and costs. Businesses need to forecast future demands for consumer products to maintain sufficient inventory of their products. Power companies need to forecast demand for electricity, to make informed purchases of energy contracts and to construct new power plants.

尽管对未来事件的预测必然是不确定的，但预测是对未来进行规划的关键部分。 网站所有者需要预测他们网站的访问量，以便提供足够的硬件资源，并预测未来的收入和成本。 企业需要预测未来对消费产品的需求，以保持足够的产品库存。 电力公司需要预测电力需求，明智地购买能源合同，并建设新的发电厂。

Methods for forecasting time series can also be applied to infer the causal impact of a feature launch or other intervention on user engagement metrics [1], to infer the current value of difficult-to-observe quantities like the unemployment rate from more readily available information [2], as well as to detect anomalies in time series data.

预测时间序列的方法也可以用来推断特征发射或其他干预对用户参与度量[1]的因果影响，从更容易获得的信息[2]中推断难以观察的数量的当前值，以及检测时间序列数据中的异常。

### Structural Time Series

### 结构时间序列

Structural time series (STS) models [3] are a family of probability models for time series that includes and generalizes many standard time-series modeling ideas, including:

结构时间序列(STS)模型[3]是一系列时间序列的概率模型，它包括并概括了许多标准的时间序列建模思想，包括:

- autoregressive processes, 自回归过程,
- moving averages, 移动平均线
- local linear trends, 局部线性趋势
- seasonality, and 季节性，以及
- regression and variable selection on external covariates (other time series potentially related to the series of interest). 外部协变量的回归和变量选择(其他时间序列可能与感兴趣的序列相关)

An STS model expresses an observed time series as the sum of simpler components:

Sts 模型将观测到的时间序列表示为简单成分之和:





The individual components are each time series governed by a particular structural assumption. For example, one component might encode a seasonal effect (e.g., day-of-week effects), another a local linear trend, and another a linear dependence on some set of covariate time series.

每个组成部分都是由一个特定的结构假设支配的时间序列。 例如，一个组成部分可能编码季节效应(例如，每周一天的效应) ，另一个组成部分编码局部线性趋势，另一个组成部分编码对某些协变量时间序列的线性依赖。

By allowing modelers to encode assumptions about the processes generating the data, structural time series can often produce reasonable forecasts from relatively little data (e.g., just a single input series with tens of points). The model’s assumptions are interpretable, and we can interpret the predictions by visualizing the decompositions of past data and future forecasts into structural components. Moreover, structural time series models use a probabilistic formulation that can naturally handle missing data and provide a principled quantification of uncertainty.

通过允许建模者对产生数据的过程进行编码假设，结构时间序列通常可以从相对较少的数据(例如，只有一个数十个点的单一输入序列)产生合理的预测。 模型的假设是可以解释的，我们可以通过将过去的数据和未来的预测可视化分解为结构成分来解释预测。 此外，结构时间序列模型使用概率公式，可以自然地处理缺失数据，并提供了原则性的不确定性量化。

### Structural Time Series in TensorFlow Probability

### 结构时间序列在 TensorFlow 概率中的应用

TensorFlow Probability (TFP) now features built-in support for fitting and forecasting using structural time series models. This support includes Bayesian inference of model parameters using variational inference (VI) and Hamiltonian Monte Carlo (HMC), computing both point forecasts and predictive uncertainties. Because they’re built in TensorFlow, these methods naturally take advantage of vectorized hardware (GPUs and TPUs), can efficiently process many time series in parallel, and can be integrated with deep neural networks.

Tensorflow 概率(TFP)现在内置支持使用结构时间序列模型进行拟合和预测。 这种支持包括使用变分推理(VI)和哈密顿蒙特卡罗(HMC)计算模型参数的贝叶斯推断，同时计算点预测和预测不确定性。 由于这些方法都是建立在 TensorFlow 上的，所以它们很自然地利用了矢量化的硬件(gpu 和 TPUs) ，能够有效地并行处理多个时间序列，并且能够与深层神经网络集成。

### Example: Forecasting CO2 Concentration

### 例如: 二氧化碳浓度的预测

To see structural time series in action, consider this monthly record of atmospheric CO2 concentration from the Mauna Loa observatory in Hawaii [5]:

为了了解结构性时间序列的作用，我们来看看这个夏威夷莫纳罗亚火山天文台空间站提供的每月大气二氧化碳浓度记录



![img](https://cdn-images-1.medium.com/max/800/0*PQ09cqE6xNJZAf5G)

It should be clear by inspection that this series contains both a long-term trend and annual seasonal variation. We can encode these two components directly in a structural time series model, using just a few lines of TFP code:

通过检查可以清楚地看出，这个系列既包含长期趋势，也包含年度季节性变化。 我们可以用一个结构化的时间序列模型直接对这两个部分进行编码，只需要使用几行 TFP 代码:

```python

import tensorflow_probability as tfp
trend = tfp.sts.LocalLinearTrend(observed_time_series=co2_by_month)
seasonal = tfp.sts.Seasonal(
    num_seasons=12, observed_time_series=co2_by_month)
model = tfp.sts.Sum([trend, seasonal], observed_time_series=co2_by_month)
```

Here we’ve used a local linear trend model, which assumes the trend is linear, with slope evolving slowly over time following a random walk. Fitting the model to the data produces a probabilistic forecast based on our modeling assumptions:

这里我们使用了一个局部线性趋势模型，它假设趋势是线性的，随着时间的推移，斜率在随机游走之后缓慢演化。 根据我们的建模假设，将模型与数据进行拟合得到一个概率预测:



![img](https://cdn-images-1.medium.com/max/800/0*Lgachj7sD9PBnfAR)

We can see that the forecast uncertainty (shading ± 2 standard deviations) increases over time, as the linear trend model becomes less confident in its extrapolation of the slope. The mean forecast combines the seasonal variational with a linear extrapolation of the existing trend, which appears to slightly underestimate the accelerating growth in atmospheric CO2, but the true values are still within the 95% predictive interval.

我们可以看到，随着时间的推移，预测的不确定性(遮蔽2个标准差)增加，因为线性趋势模型对其斜率外推的信心变得越来越差。 平均预报结合了季节变化和现有趋势的线性外推，这似乎略微低估了大气 CO2的加速增长，但真实值仍然在95% 的预测区间内。

The [full code for this example](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/jupyter_notebooks/Structural_Time_Series_Modeling_Case_Studies_Atmospheric_CO2_and_Electricity_Demand.ipynb) is available on Github.

这个例子的完整代码可以在 Github 上找到。

### Example: Forecasting Demand for Electricity

### 例子: 电力需求预测

Next we’ll consider a more complex example: forecasting electricity demand in Victoria, Australia. The top line of this plot shows an hourly record from the first six weeks of 2014 (data from [4], available at [https://github.com/robjhyndman/fpp2-package):](https://github.com/robjhyndman/fpp2-package%29:)

接下来我们将考虑一个更复杂的例子: 预测澳大利亚维多利亚的电力需求。 这张图的顶线显示了2014年前六个星期的每小时记录(数据来自[4] ，可在 https://github.com/robjhyndman/fpp2-package 获得) :



![img](https://cdn-images-1.medium.com/max/800/0*MKIeUw0BDmjLhGX_)

Here we have access to an external source of information: the temperature, which correlates with electrical demand for air conditioning. Remember that January is summer in Australia! Let’s incorporate this temperature data in a STS model, which can include external covariates via linear regression:

在这里，我们可以访问外部信息来源: 温度，这与空气调节的电力需求有关。 记住，一月是澳大利亚的夏天！ 让我们把这些温度数据整合到一个 STS 模型中，这个模型可以通过线性回归来包含外部协变量:



<iframe width="700" height="250" data-src="/media/a5b96322ddf96550cdada23445f62c21?postId=344edac24083" data-media-id="a5b96322ddf96550cdada23445f62c21" data-thumbnail="https://i.embed.ly/1/image?url=https%3A%2F%2Favatars1.githubusercontent.com%2Fu%2F991882%3Fs%3D400%26v%3D4&amp;key=a19fcc184b9711e1b4764040d3dc5c07" class="progressiveMedia-iframe js-progressiveMedia-iframe" allowfullscreen="" frameborder="0" src="https://medium.com/media/a5b96322ddf96550cdada23445f62c21?postId=344edac24083" style="display: block; position: absolute; margin: auto; max-width: 100%; box-sizing: border-box; transform: translateZ(0px); top: 0px; left: 0px; width: 700px; height: 483px;"></iframe>

Note that we’ve also included multiple seasonality effects: an hour-of-day, a day-of-week effect, and an autoregressive component to model any unexplained residual effects. We could have used a simple random walk, but chose an autoregressive component because it maintains bounded variance over time.

请注意，我们还包括了多种季节性效应: 一小时一天的效应，一周一天的效应，以及一个自回归组件来建模任何无法解释的残余效应。 我们可以使用一个简单的随机游动，但选择一个自回归分量，因为它随着时间维持有界方差。



![img](https://cdn-images-1.medium.com/max/800/0*IlrfzF-A1tkCVGO8)

The forecast from this model isn’t perfect — there are apparently still some unmodeled sources of variation — but it’s not crazy, and again the uncertainties look reasonable. We can better understand this forecast by visualizing the decomposition into components (note that each component plot has a different y-axis scale):

这个模型的预测并不完美ーー显然还有一些未建模的变化来源ーー但这并不疯狂，而且这些不确定性看起来也是合理的。 我们可以通过可视化分解成组件(注意每个组件图都有不同的 y 轴尺度)来更好地理解这个预测:



![img](https://cdn-images-1.medium.com/max/800/0*on1ndBZpDc1-rBgG)

We see that the model has quite reasonably identified a large hour-of-day effect and a much smaller day-of-week effect (the lowest demand appears to occur on Saturdays and Sundays), as well as a sizable effect from temperature, and that it produces relatively confident forecasts of these effects. Most of the predictive uncertainty comes from the autoregressive process, based on its estimate of the unmodeled (residual) variation in the observed series.

我们看到，该模型已经相当合理地确定了一个大的小时效应和一个小得多的每周一天效应(最低的需求似乎发生在星期六和星期日) ，以及一个相当大的温度效应，并且它产生了对这些效应的相对自信的预测。 大多数的预测不确定性来自于自回归过程，基于它对观测序列中未建模(残差)变化的估计。

A modeler might use this decomposition to understand how to improve the model. For example, they might notice that some spikes in temperature still seem to coincide with spikes in the AR residual, indicating that additional features or data transformations might help better capture the temperature effect.

建模者可能会使用这种分解来理解如何改进模型。 例如，他们可能会注意到，一些温度峰值似乎仍然与 AR 残差峰值重合，这表明额外的特征或数据转换可能有助于更好地捕捉温度效应。

The [full code for this example](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/jupyter_notebooks/Structural_Time_Series_Modeling_Case_Studies_Atmospheric_CO2_and_Electricity_Demand.ipynb) is available on Github.

这个例子的完整代码可以在 Github 上找到。

### The TensorFlow Probability STS Library

### Tensorflow 概率 STS 库

As the above examples show, STS models in TFP are built by adding together model components. STS provides modeling components like:

如上面的示例所示，TFP 中的 STS 模型是通过将模型组件添加到一起来构建的。 Sts 提供了如下建模组件:

- [Autoregressive 自回归](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/Autoregressive), [LocalLinearTrend 定位到耳朵趋势](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/LocalLinearTrend), [SemiLocalLinearTread 深圳市半月线贸易有限公司](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/SemiLocalLinearTrend), and ，及[LocalLevel](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/LocalLevel). For modeling time series with a level or slope that evolves according to a random walk or other process. . 用于建模具有水平或斜率的时间序列，该时间序列根据随机漫步或其他过程进化
- [Seasonal 季节性的](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/Seasonal). For time series depending on seasonal factors, such as the hour of the day, the day of the week, or the month of the year. . 取决于季节因素的时间序列，如一天的小时，一周的天，或一年的月份
- [LinearRegression 线性回归](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/LinearRegression). For time series depending on additional, time-varying covariates. Regression components can also be used to encode holiday or other date-specific effects. . 对于依赖于附加的时变协变量的时间序列。 回归分量也可用于编码假日或其他特定日期的效果

STS provides methods for fitting the resulting time series models with [variational inference](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/build_factored_variational_loss) and [Hamiltonian Monte Carlo](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/fit_with_hmc).

Sts 提供了用变分推理和哈密顿蒙特卡罗方法拟合时间序列模型的方法。

Check out our code, documentation, and further examples on [the TFP home page](https://www.tensorflow.org/probability/).

在 TFP 主页上查看我们的代码、文档和进一步的示例。

Structural time series are being used for several important time series applications inside Google. We hope you will find them useful, as well. Please join the [tfprobability@tensorflow.org ](https://groups.google.com/a/tensorflow.org/forum/#!forum/tfprobability)forum for the latest Tensorflow Probability announcements and other TFP discussions!

结构时间序列在谷歌内部被用于几个重要的时间序列应用。 我们希望你会发现它们也是有用的。 请加入 tfprobability@Tensorflow. org 论坛，了解最新的 Tensorflow 概率公告和其他 TFP 讨论！

### References

### 参考资料

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

