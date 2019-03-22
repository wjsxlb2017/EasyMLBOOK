# 使用基于tf.Probability库的结构化时序模型进行时序预测
#

- [ ] 写总结 
## Overview
> *“It is difficult to make predictions, especially about the future.”*
> *—  * [Karl Kristian Steincke](https://quoteinvestigator.com/2013/10/20/no-predict/) 
> Although predictions of future events are necessarily uncertain, forecasting is a critical part of planning for the future. Website owners need to forecast the number of visitors to their site in order to provision sufficient hardware resources, as well as predict future revenue and costs. Businesses need to forecast future demands for consumer products to maintain sufficient inventory of their products. Power companies need to forecast demand for electricity, to make informed purchases of energy contracts and to construct new power plants.
> Methods for forecasting time series can also be applied to infer the causal impact of a feature launch or other intervention on user engagement metrics [1], to infer the current value of difficult-to-observe quantities like the unemployment rate from more readily available information [2], as well as to detect anomalies in time series data.

## Structural TimeSeries
**Structural time series (STS) models** [3] are a family of probability models for time series that includes and generalizes many standard time-series modeling ideas, including:
* autoregressive processes,
* moving averages,
* local linear trends,
* seasonality, and
* regression and variable selection on external covariates (other time series potentially related to the series of interest).
  An STS model expresses an observed time series as the sum of simpler components:

[image:22322DD3-9C3D-4976-8AAA-A990C07EF2C4-7872-0000790B3DB76FDF/1*WxTag1CSij-GlhYvGh3Mjw.png]
The individual components are each time series governed by a particular structural assumption. For example, one component might encode a seasonal effect (e.g., day-of-week effects), another a local linear trend, and another a linear dependence on some set of covariate time series.
By allowing modelers to encode assumptions about the processes generating the data, structural time series can often produce reasonable forecasts from relatively little data (e.g., just a single input series with tens of points). The model’s assumptions are interpretable, and we can interpret the predictions by visualizing the decompositions of past data and future forecasts into structural components. Moreover, structural time series models use a probabilistic formulation that can naturally handle missing data and provide a principled quantification of uncertainty.
## Structural Time Series in TensorFlow Probability
TensorFlow Probability (TFP) now features built-in support for fitting and forecasting using structural time series models. This support includes Bayesian inference of model parameters using variational inference (VI) and Hamiltonian Monte Carlo (HMC), computing both point forecasts and predictive uncertainties. Because they’re built in TensorFlow, these methods naturally take advantage of vectorized hardware (GPUs and TPUs), can efficiently process many time series in parallel, and can be integrated with deep neural networks.
## Example: 
Forecasting CO2 ConcentrationTo see structural time series in action, consider this monthly record of atmospheric CO2 concentration from the Mauna Loa observatory in Hawaii [5]:


[image:74144B2F-E7CA-4131-AE0B-8B46F5A23488-7872-0000790B3D29FE7B/0*PQ09cqE6xNJZAf5G.png]
It should be clear by inspection that this series contains both a long-term trend and annual seasonal variation. We can encode these two components directly in a structural time series model, using just a few lines of TFP code:


Here we’ve used a local linear trend model, which assumes the trend is linear, with slope evolving slowly over time following a random walk. Fitting the model to the data produces a probabilistic forecast based on our modeling assumptions:


[image:659FF33F-A6B2-442C-9AA6-498B258FEB13-7872-0000790B3CD6928E/0*Lgachj7sD9PBnfAR.png]
We can see that the forecast uncertainty (shading ± 2 standard deviations) increases over time, as the linear trend model becomes less confident in its extrapolation of the slope. The mean forecast combines the seasonal variational with a linear extrapolation of the existing trend, which appears to slightly underestimate the accelerating growth in atmospheric CO2, but the true values are still within the 95% predictive interval.
The [full code for this example](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/jupyter_notebooks/Structural_Time_Series_Modeling_Case_Studies_Atmospheric_CO2_and_Electricity_Demand.ipynb) is available on Github.
## Example:
 Forecasting Demand for ElectricityNext we’ll consider a more complex example: forecasting electricity demand in Victoria, Australia. The top line of this plot shows an hourly record from the first six weeks of 2014 (data from [4], available at [https://github.com/robjhyndman/fpp2-package):](https://github.com/robjhyndman/fpp2-package%29:) 


[image:D352EA94-B60F-465C-B155-FC3EEFE4D584-7872-0000790B3C4C1EB8/0*MKIeUw0BDmjLhGX_.png]
Here we have access to an external source of information: the temperature, which correlates with electrical demand for air conditioning. Remember that January is summer in Australia! Let’s incorporate this temperature data in a STS model, which can include external covariates via linear regression:



Note that we’ve also included multiple seasonality effects: an hour-of-day, a day-of-week effect, and an autoregressive component to model any unexplained residual effects. We could have used a simple random walk, but chose an autoregressive component because it maintains bounded variance over time.


[image:190B73C2-573E-441F-9D16-42AC849203CB-7872-0000790B3BFBAD07/0*IlrfzF-A1tkCVGO8.png]
The forecast from this model isn’t perfect — there are apparently still some unmodeled sources of variation — but it’s not crazy, and again the uncertainties look reasonable. We can better understand this forecast by visualizing the decomposition into components (note that each component plot has a different y-axis scale):


[image:6ABF8BA8-1038-4674-A85D-8C1536E8BAB3-7872-0000790B3B87C5F9/0*on1ndBZpDc1-rBgG.png]
We see that the model has quite reasonably identified a large hour-of-day effect and a much smaller day-of-week effect (the lowest demand appears to occur on Saturdays and Sundays), as well as a sizable effect from temperature, and that it produces relatively confident forecasts of these effects. Most of the predictive uncertainty comes from the autoregressive process, based on its estimate of the unmodeled (residual) variation in the observed series.
A modeler might use this decomposition to understand how to improve the model. For example, they might notice that some spikes in temperature still seem to coincide with spikes in the AR residual, indicating that additional features or data transformations might help better capture the temperature effect.
The [full code for this example](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples/jupyter_notebooks/Structural_Time_Series_Modeling_Case_Studies_Atmospheric_CO2_and_Electricity_Demand.ipynb) is available on Github.

## The TensorFlow Probability STSLibrary
As the above examples show, STS models in TFP are built by adding together model components. STS provides modeling components like:
*  [Autoregressive](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/Autoregressive) , [LocalLinearTrend](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/LocalLinearTrend) , [SemiLocalLinearTread](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/SemiLocalLinearTrend) , and [LocalLevel](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/LocalLevel) . For modeling time series with a level or slope that evolves according to a random walk or other process.
*  [Seasonal](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/Seasonal) . For time series depending on seasonal factors, such as the hour of the day, the day of the week, or the month of the year.
*  [LinearRegression](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/LinearRegression) . For time series depending on additional, time-varying covariates. Regression components can also be used to encode holiday or other date-specific effects.
  STS provides methods for fitting the resulting time series models with [variational inference](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/build_factored_variational_loss) and [Hamiltonian Monte Carlo](https://www.tensorflow.org/probability/api_docs/python/tfp/sts/fit_with_hmc) .
  Check out our code, documentation, and further examples on [the TFP home page](https://www.tensorflow.org/probability/) .
  Structural time series are being used for several important time series applications inside Google. We hope you will find them useful, as well. Please join the [tfprobability@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/tfprobability) forum for the latest Tensorflow Probability announcements and other TFP discussions!

# References
[1] Brodersen, K. H., Gallusser, F., Koehler, J., Remy, N., & Scott, S. L. (2015). Inferring causal impact using Bayesian structural time-series models.*The Annals of Applied Statistics*,*9*(1), 247–274.
[2] Choi, H., & Varian, H. (2012). Predicting the present with Google Trends. Economic Record, 88, 2–9.
[3] Harvey, A. C. (1989).*Forecasting, structural time series models and the Kalman filter*. Cambridge University Press.
[4] Hyndman, R.J., & Athanasopoulos, G. (2018). Forecasting: principles and practice, 2nd edition, OTexts: Melbourne, Australia. OTexts.com/fpp2. Accessed on February 23, 2019.
[5] Keeling, C. D., Piper, S. C., Bacastow, R. B., Wahlen, M., Whorf, T. P., Heimann, M., & Meijer, H. A. (2001). Exchanges of atmospheric CO2 and 13CO2 with the terrestrial biosphere and oceans from 1978 to 2000. I. Global aspects, SIO Reference Series, №01–06, Scripps Institution of Oceanography, San Diego.

*  [Time Series](https://medium.com/tag/timeseries?source=post)  [Bsts](https://medium.com/tag/bst?source=post)  [Tensorflow Probability](https://medium.com/tag/tensorflow-probability?source=post)  [Sts](https://medium.com/tag/sts?source=post) 