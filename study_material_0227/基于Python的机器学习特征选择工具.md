

[TOC]



![img](https://cdn-images-1.medium.com/max/2000/1*rqeNwjiakRS-SqsW9wCgrA.jpeg)

# 基于 Python 的机器学习特征选择工具

## 使用 FeatureSelector 实现高效的机器学习工作流

Feature selection, the process of finding and selecting the most useful features in a dataset, is a crucial step of the machine learning pipeline. Unnecessary features decrease training speed, decrease model interpretability, and, most importantly, decrease generalization performance on the test set.

特征选择是机器学习过程中的一个关键步骤，是在数据集中寻找和选择最有用的特征的过程。 不必要的特征会降低训练速度，降低模型的可解释性，最重要的是，会降低测试集的泛化性能。

Frustrated by the ad-hoc feature selection methods I found myself applying over and over again for machine learning problems, I built a class for feature selection in Python [available on GitHub](https://github.com/WillKoehrsen/feature-selector). The `FeatureSelector` includes some of the most common feature selection methods:

我发现自己一遍又一遍地应用特定的特征选择方法来解决机器学习问题，这让我感到沮丧，于是我在 GitHub 上构建了一个 Python 特征选择类。 功能选择器包括一些最常见的特征选择方法:

1. **Features with a high percentage of missing values** 缺失值百分比高的特性
2. **Collinear (highly correlated) features** 共线(高度相关)特征
3. **Features with zero importance in a tree-based model** 基于树模型的零重要特征
4. **Features with low importance** 低重要性的特征
5. **Features with a single unique value** 具有单个唯一值的特性

In this article we will walk through using the `FeatureSelector` on an example machine learning dataset. We’ll see how it allows us to rapidly implement these methods, allowing for a more efficient workflow.

在本文中，我们将在一个示例机器学习数据集上使用 FeatureSelector。 我们将看到它如何允许我们快速实现这些方法，从而实现更高效的工作流。

The complete code is [available on GitHub](https://github.com/WillKoehrsen/feature-selector) and I encourage any contributions. The Feature Selector is a work in progress and will continue to improve based on the community needs!

完整的代码可以在 GitHub 上找到，我鼓励任何贡献。 功能选择器是一个正在进行的工作，并将继续改善基于社区的需求！

------

#### 数据集示例

For this example, we will use a sample of data from the [Home Credit Default Risk machine learning competition](https://www.kaggle.com/c/home-credit-default-risk) on Kaggle. (To get started with the competition, see [this article](https://towardsdatascience.com/machine-learning-kaggle-competition-part-one-getting-started-32fb9ff47426)). The entire dataset is [available for download](https://www.kaggle.com/c/home-credit-default-risk/data) and here we will use a sample for illustration purposes.

对于这个例子，我们将使用来自 Kaggle 上举办的 Home Credit Default Risk 机器学习竞赛的数据样本。 (为了开始比赛，请看这篇文章)。 整个数据集可供下载，这里我们将使用一个示例进行说明。



![img](https://cdn-images-1.medium.com/max/2000/1*W0qSMsheaWsXJBJ7i2pH4g.png)

Example data. TARGET is the label for classification 示例数据。 Target 是用于分类的标签

The competition is a supervised classification problem and this is a good dataset to use because it has many missing values, numerous highly correlated (collinear) features, and a number of irrelevant features that do not help a machine learning model.

这个竞赛是一个监督分类问题，这是一个很好的数据集，因为它有许多缺失值，许多高度相关(共线)的特征，和一些不相关的特征，不利于机器学习模型。

#### Creating an Instance

#### 创建一个实例

To create an instance of the `FeatureSelector` class, we need to pass in a structured dataset with observations in the rows and features in the columns. We can use some of the methods with only features, but the importance-based methods also require training labels. Since we have a supervised classification task, we will use a set of features and a set of labels.

要创建 FeatureSelector 类的实例，我们需要传入一个结构化的数据集，其中包含行中的观察值和列中的特性。 我们可以使用一些只有特征的方法，但是基于重要性的方法也需要训练标签。 由于我们有一个监督分类任务，我们将使用一组特征和一组标签。

(Make sure to run this in the same directory as `feature_selector.py` )

(确保在与 feature selector.py 相同的目录中运行此命令)

```
from feature_selector import FeatureSelector
# Features are in train and labels are in train_labels
fs = FeatureSelector(data = train, labels = train_labels)
```

#### Methods

#### 方法

The feature selector has five methods for finding features to remove. We can access any of the identified features and remove them from the data manually, or use the `remove` function in the Feature Selector.

特性选择器有五种查找要删除的特性的方法。 我们可以访问任何已识别的特性并手动从数据中删除它们，或者使用 Feature Selector 中的 remove 函数。

Here we will go through each of the identification methods and also show how all 5 can be run at once. The `FeatureSelector` additionally has several plotting capabilities because visually inspecting data is a crucial component of machine learning.

在这里，我们将通过每个标识方法，并显示如何所有5可以一次运行。 另外，由于视觉检测数据是机器学习的关键组成部分，FeatureSelector 还具有几个绘图功能。

### Missing Values

### 缺失值

The first method for finding features to remove is straightforward: find features with a fraction of missing values above a specified threshold. The call below identifies features with more than 60% missing values (**bold** is output).

查找要删除的特征的第一种方法很简单: 查找缺失值小于指定阈值的特征。 下面的调用标识缺失值超过60% 的特性(粗体表示输出)。

```
fs.identify_missing(missing_threshold = 0.6)
17 features with greater than 0.60 missing values.
```

We can see the fraction of missing values in every column in a dataframe:

我们可以在一个数据框中看到每一列中缺失值的部分:

```
fs.missing_stats.head()
```



![img](https://cdn-images-1.medium.com/max/1600/1*fpLJQBGZWhQXPFG5FyA1kg.png)

To see the features identified for removal, we access the `ops` attribute of the `FeatureSelector` , a Python dict with features as lists in the values.

为了查看要删除的特征，我们访问 FeatureSelector 的 ops 属性，这是一个 Python，它的特性是值中的列表。

```
missing_features = fs.ops['missing']
missing_features[:5]
['OWN_CAR_AGE',
 'YEARS_BUILD_AVG',
 'COMMONAREA_AVG',
 'FLOORSMIN_AVG',
 'LIVINGAPARTMENTS_AVG']
```

Finally, we have a plot of the distribution of missing values in all feature:

最后，我们绘制了所有特征中缺失值的分布图:

```
fs.plot_missing()
```



![img](https://cdn-images-1.medium.com/max/1600/1*0WBIKN83twXyWfyx9LG7Qg.png)

### Collinear Features

### 共线性特征

[Collinear features](https://www.quora.com/Why-is-multicollinearity-bad-in-laymans-terms-In-feature-selection-for-a-regression-model-intended-for-use-in-prediction-why-is-it-a-bad-thing-to-have-multicollinearity-or-highly-correlated-independent-variables) are features that are highly correlated with one another. In machine learning, these lead to decreased generalization performance on the test set due to high variance and less model interpretability.

共线特征是彼此高度相关的特征。 在机器学习中，由于高方差和较少的模型可解释性，这些导致测试集上的泛化性能降低。

The `identify_collinear` method finds collinear features based on a specified [correlation coefficient](http://www.statisticshowto.com/probability-and-statistics/correlation-coefficient-formula/) value. For each pair of correlated features, it identifies one of the features for removal (since we only need to remove one):

identify_collinear方法通过指定的相关系数阈值找到共线性特征。 对于每一对线性相关的特征对，它确定其中一个要移除的特征(因为我们只需要移除一个) :

```
fs.identify_collinear(correlation_threshold = 0.98)
21 features with a correlation magnitude greater than 0.98.
```

A neat visualization we can make with correlations is a heatmap. This shows all the features that have at least one correlation above the threshold:

关于相关性，我们可以做出一个简洁的可视化图像，这就是热度图。 这里显示了至少有一个相关性高于阈值的所有特性:

```
fs.plot_collinear()
```



![img](https://cdn-images-1.medium.com/max/1600/1*_gK6g3YWylcgfL5Bz8JMUg.png)

As before, we can access the entire list of correlated features that will be removed, or see the highly correlated pairs of features in a dataframe.

与前面一样，我们可以访问将被删除的相关特性的整个列表，或者在数据框中查看高度相关的特性对。

```
# list of collinear features to remove
collinear_features = fs.ops['collinear']
# dataframe of collinear features
fs.record_collinear.head()
```



![img](https://cdn-images-1.medium.com/max/1600/1*unCzyN2BgucGodbioUz-Kw.png)

If we want to investigate our dataset, we can also make a plot of all the correlations in the data by passing in `plot_all = True` to the call:

如果我们想要调查我们的数据集，我们也可以通过传入 plot all True to the call 来绘制数据中所有相关性的图:



![img](https://cdn-images-1.medium.com/max/1600/1*fcLsRYskgzWxVoxj4npfvg.png)

### Zero Importance Features

### 零重要性特征

The previous two methods can be applied to any structured dataset and are **deterministic** — the results will be the same every time for a given threshold. The next method is designed only for supervised machine learning problems where we have labels for training a model and is non-deterministic. The `identify_zero_importance `function finds features that have zero importance according to a gradient boosting machine (GBM) learning model.

前两种方法可以应用于任何结构化数据集，并且是确定性的ー对于给定的阈值，每次得到的结果都是相同的。 下一个方法是专门为监督式学习问题设计的，在这些问题中我们有用于训练模型的标签，并且是不确定的。 identify_zero_importance函数根据梯度提升机器学习模型找到零重要性的特征。

With tree-based machine learning models, [such as a boosting ensemble, we can find feature importances.](https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/) The absolute value of the importance is not as important as the relative values, which we can use to determine the most relevant features for a task. We can also use feature importances for feature selection by removing zero importance features. In a tree-based model, the [features with zero importance are not used to split any nodes](https://www.salford-systems.com/blog/dan-steinberg/what-is-the-variable-importance-measure), and so we can remove them without affecting model performance.

使用基于树的机器学习模型，例如 boosting 集成，我们可以发现特征的重要性。 重要性的绝对值不如相对值重要，因为我们可以用相对值来确定任务最相关的特征。 我们也可以通过去除零重要性特征来使用特征重要性进行特征选择。 在基于树的模型中，零重要性特征不用于分割任何节点，因此我们可以在不影响模型性能的情况下删除它们。

The `FeatureSelector` finds feature importances using the gradient boosting machine from the [LightGBM library](http://lightgbm.readthedocs.io/). The feature importances are averaged over 10 training runs of the GBM in order to reduce variance. Also, the model is trained using early stopping with a validation set (there is an option to turn this off) to prevent overfitting to the training data.

使用 LightGBM 库的GBM的FeatureSelector能找到特征的重要性。 为了减少方差，对 GBM 的10次训练次数平均特征输入。 此外，使用验证集的早期停止训练模型(可以关闭此选项) ，以防止对训练数据过度拟合。

The code below calls the method and extracts the zero importance features:

下面的代码调用该方法并提取出零重要性特征:

```
# Pass in the appropriate parameters
fs.identify_zero_importance(task = 'classification', 
                            eval_metric = 'auc', 
                            n_iterations = 10, 
                             early_stopping = True)
# list of zero importance features
zero_importance_features = fs.ops['zero_importance']
63 features with zero importance after one-hot encoding.
```

The parameters we pass in are as follows:

我们传入的参数如下:

- `task` : either “classification” or “regression” corresponding to our problem : 与我们的问题相对应的"分类"或"回归"
- `eval_metric`: metric to use for early stopping (not necessary if early stopping is disabled) : 公制用于提前停止(如果禁用提前停止，则无需提前停止)
- `n_iterations` : number of training runs to average the feature importances over : 训练次数平均特征重要性超过
- `early_stopping`: whether or not use early stopping for training the model : 是否使用提前停止训练模型

This time we get two plots with `plot_feature_importances`:

这一次，我们得到了两个情节特征重要的情节:

```
# plot the feature importances
fs.plot_feature_importances(threshold = 0.99, plot_n = 12)
124 features required for 0.99 of cumulative importance
```



![img](https://cdn-images-1.medium.com/max/1200/1*hWCOAEWkH4z5BKKqkFAd1g.png)



![img](https://cdn-images-1.medium.com/max/1200/1*HJk89EkbcmriiWbxpV6Uew.png)

On the left we have the `plot_n` most important features (plotted in terms of normalized importance where the total sums to 1). On the right we have the cumulative importance versus the number of features. The vertical line is drawn at `threshold` of the cumulative importance, in this case 99%.

在左边我们有图 n 个最重要的特征(按归一化的重要性绘制，总和为1)。 在右边我们有累积的重要性与功能的数量。 垂直线在累积重要性的阈值处画出，在本例中为99% 。

Two notes are good to remember for the importance-based methods:

对于以重要性为基础的方法，有两个注意事项值得记住:

- Training the gradient boosting machine is stochastic meaning the 训练梯度提升机器是随机的，这意味着*feature importances will change every time the model is run 特性的重要性会随着模型的运行而改变*

This should not have a major impact (the most important features will not suddenly become the least) but it will change the ordering of some of the features. It also can affect the number of zero importance features identified. Don’t be surprised if the feature importances change every time!

这不应该产生重大影响(最重要的特征不会突然变成最不重要的) ，但它会改变一些特征的顺序。 它也可以影响零重要性特征识别的数量。 如果特征的重要性每次都改变，不要感到惊讶！

- To train the machine learning model, the features are first 为了对机器学习模型进行训练，特征是第一位的*one-hot encoded One-hot 编码*. This means some of the features identified as having 0 importance might be one-hot encoded features added during modeling. . 这意味着一些被识别为具有0重要性的特征可能是在建模期间添加的一个*One-hot 编码*特征

When we get to the feature removal stage, there is an option to remove any added one-hot encoded features. However, if we are doing machine learning after feature selection, we will have to one-hot encode the features anyway!

当我们进入特征删除阶段时，有一个选项可以删除任何添加的one-hot编码特征。 然而，如果我们在特征选择之后做机器学习后，无论如何，我们将不得不对特征进行one-hot编码！

### Low Importance Features

### 低重要性特征

The next method builds on zero importance function, using the feature importances from the model for further selection. The function `identify_low_importance` finds the lowest importance features that do not contribute to a specified total importance.

第二种方法建立在零重要性函数的基础上，利用模型中的特征输入进行进一步的选择。 识别低重要性的功能找出对特定的总重要性没有贡献的最低重要性的特征。

For example, the call below finds the least important features that are not required for achieving 99% of the total importance:

例如，下面的调用找到了不需要达到总重要性的99% 的最不重要的特性:

```
fs.identify_low_importance(cumulative_importance = 0.99)
123 features required for cumulative importance of 0.99 after one hot encoding.
116 features do not contribute to cumulative importance of 0.99.
```

Based on the plot of cumulative importance and this information, the gradient boosting machine considers many of the features to be irrelevant for learning. Again, the results of this method will change on each training run.

基于累积重要性的情节和这些信息，梯度提升机器认为许多特征对于学习是无关的。 同样，此方法的结果将在每次训练运行中更改。

To view all the feature importances in a dataframe:

查看dataframe中所有特征的重要性:

```
fs.feature_importances.head(10)
```



![img](https://cdn-images-1.medium.com/max/1600/1*d1uRrw212LAmpjlszj7CFg.png)

The `low_importance` method borrows from one of the methods of [using Principal Components Analysis (PCA) ](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)where it is common to keep only the PC needed to retain a certain percentage of the variance (such as 95%). The percentage of total importance accounted for is based on the same idea.

低重要性方法借鉴了使用主成分分析(PCA)的一种方法，在这种方法中，通常只保留 PC 需要保留一定百分比的方差(如95%)。 占总重要性的百分比也是基于同样的想法。

The feature importance based methods are really only applicable if we are going to use a tree-based model for making predictions. Besides being stochastic, the importance-based methods are a black-box approach in that we don’t really know why the model considers the features to be irrelevant. If using these methods, run them several times to see how the results change, and perhaps create multiple datasets with different parameters to test!

基于特征重要性的方法只有在我们使用基于树的模型进行预测时才真正适用。 除了具有随机性之外，基于重要性的方法是一种黑箱方法，因为我们并不真正知道为什么模型认为特征是无关的。 如果使用这些方法，请多次运行它们以查看结果如何更改，并可能创建具有不同参数的多个数据集来进行测试！

### Single Unique Value Features

### 单一独特价值功能

The final method is fairly basic: [find any columns that have a single unique value.](https://github.com/Featuretools/featuretools/blob/master/featuretools/selection/selection.py) A feature with only one unique value cannot be useful for machine learning because this [feature has zero variance](https://www.r-bloggers.com/near-zero-variance-predictors-should-we-remove-them/). For example, a tree-based model can never make a split on a feature with only one value (since there are no groups to divide the observations into).

最后一个方法非常简单: 查找任何具有单个唯一值的列。 只有一个唯一值的特征对机器学习没有用处，因为这个特征的方差为零。 例如，一个基于树的模型永远不能对一个只有一个值的特性进行分割(因为没有可以将观察值分割的组)。

There are no parameters here to select, unlike the other methods:

这里没有参数可供选择，不像其他方法:

```
fs.identify_single_unique()
4 features with a single unique value.
```

We can plot a histogram of the number of unique values in each category:

我们可以绘制每个类别中唯一值的数量直方图:

```
fs.plot_unique()
```



![img](https://cdn-images-1.medium.com/max/1600/1*F3BV5mUWG-GLP8gnS62Z6w.png)

One point to remember is `NaNs` are dropped before [calculating unique values in Pandas by default.](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.nunique.html)

需要记住的一点是，在计算 Pandas 中的默认唯一值之前，会删除 NaNs。

### Removing Features

### 移除功能

Once we’ve identified the features to discard, we have two options for removing them. All of the features to remove are stored in the `ops` dict of the `FeatureSelector` and we can use the lists to remove features manually. Another option is to use the `remove` built-in function.

一旦我们确定了要丢弃的特性，我们就有两个选项来删除它们。 所有要删除的功能都存储在功能选择器的操作类中，我们可以使用列表手动删除功能。 另一个选择是使用 remove 内置函数。

For this method, we pass in the `methods` to use to remove features. If we want to use all the methods implemented, we just pass in `methods = 'all'`.

对于此方法，我们传入用于删除特性的方法。 如果我们想要使用所有已实现的方法，我们只需要传入'all'方法。

```
# Remove the features from all methods (returns a df)
train_removed = fs.remove(methods = 'all')
['missing', 'single_unique', 'collinear', 'zero_importance', 'low_importance'] methods have been run

Removed 140 features.
```

This method returns a dataframe with the features removed. To also remove the one-hot encoded features that are created during machine learning:

这个方法返回一个删除了特性的数据框。 还要去掉机器学习过程中产生的一个热门编码特性:

```
train_removed_all = fs.remove(methods = 'all', keep_one_hot=False)
Removed 187 features including one-hot features.
```

It might be a good idea to check the features that will be removed before going ahead with the operation! The original dataset is stored in the `data`attribute of the `FeatureSelector` as a back-up!

在继续操作之前，最好先检查将被删除的特性！ 原始数据集存储在 FeatureSelector 的数据属性中作为备份！

### Running all Methods at Once

### 同时运行所有方法

Rather than using the methods individually, we can use all of them with `identify_all`. This takes a dictionary of the parameters for each method:

与其单独使用这些方法，我们可以使用所有的方法来识别所有的方法。 这需要每个方法的参数字典:

```
fs.identify_all(selection_params = {'missing_threshold': 0.6,    
                                    'correlation_threshold': 0.98, 
                                    'task': 'classification',    
                                    'eval_metric': 'auc', 
                                    'cumulative_importance': 0.99})
151 total features out of 255 identified for removal after one-hot encoding.
```

Notice that the number of total features will change because we re-ran the model. The `remove` function can then be called to discard these features.

请注意，由于我们重新运行了模型，总的特性数量将会发生变化。 然后可以调用 remove 函数放弃这些特性。

### Conclusions

### 结论

The Feature Selector class implements several common [operations for removing features](https://machinelearningmastery.com/an-introduction-to-feature-selection/) before training a machine learning model. It offers functions for identifying features for removal as well as visualizations. Methods can be run individually or all at once for efficient workflows.

在训练机器学习模型之前，特征选择器类实现了几个常见的移除特征的操作。 它提供了识别要移除的特征以及可视化的功能。 方法可以单独运行，也可以一次运行所有方法，以提高工作流的效率。

The `missing`, `collinear`, and `single_unique` methods are deterministic while the feature importance-based methods will change with each run. Feature selection, much like the [field of machine learning, is largely empirical](https://hips.seas.harvard.edu/blog/2012/12/24/the-empirical-science-of-machine-learning-evaluating-rbms/)and requires testing multiple combinations to find the optimal answer. It’s best practice to try several configurations in a pipeline, and the Feature Selector offers a way to rapidly evaluate parameters for feature selection.

缺失的、共线的和唯一的方法是确定的，而基于特征重要性的方法将随着每次运行而改变。 特征选择，就像机器学习领域一样，在很大程度上是经验性的，需要测试多种组合才能找到最佳答案。 在一个管道中尝试多种配置是最佳实践，而且特征选择器提供了一种快速评估特征选择参数的方法。

As always, I welcome feedback and constructive criticism. I want to emphasis that I’m looking for help on the `FeatureSelector`. Anyone can [contribute on GitHub](https://github.com/WillKoehrsen/feature-selector) and I appreciate advice from those who just uses the tool! I can also be reached on Twitter [@koehrsen_will](http://twitter.com/@koehrsen_will).

一如既往，我欢迎反馈和建设性的批评。 我想强调的是，我正在寻求特性选择器的帮助。 任何人都可以在 GitHub 上做出贡献，我非常感谢那些只是使用这个工具的人的建议！ 我也可以通过 Twitter@koehrsen will 联系到。



- [Machine Learning 机器学习](https://towardsdatascience.com/tagged/machine-learning?source=post)
- [Python 巨蟒](https://towardsdatascience.com/tagged/python?source=post)
- [Education 教育](https://towardsdatascience.com/tagged/education?source=post)
- [Programming 编程](https://towardsdatascience.com/tagged/programming?source=post)
- [Towards Data Science 走向数据科学](https://towardsdatascience.com/tagged/towards-data-science?source=post)



Applause from you, 你们的掌声[Dipanjan (DJ) Sarkar](https://medium.com/@dipanzan.sarkar), and 及1,197 others 其他1197人

22

Follow 跟着

[![Go to the profile of Will Koehrsen](https://cdn-images-1.medium.com/fit/c/120/120/1*SckxdIFfjlR-cWXkL5ya-g.jpeg)](https://towardsdatascience.com/@williamkoehrsen?source=footer_card)

### [Will Koehrsen威尔 · 科尔森](https://towardsdatascience.com/@williamkoehrsen)

Data Scientist at Cortex Intel, Data Science Communicator

英特尔大脑皮层数据科学家，数据科学传播者

Following 随后

[![Towards Data Science](https://cdn-images-1.medium.com/fit/c/120/120/1*F0LADxTtsKOgmPa-_7iUEQ.jpeg)](https://towardsdatascience.com/?source=footer_card)

### [Towards Data Science 走向数据科学](https://towardsdatascience.com/?source=footer_card)

Sharing concepts, ideas, and codes.

分享概念、想法和代码。





More from Towards Data Science 更多关于数据科学

Mastering the Data Science Interview 掌握数据科学访谈

[![Go to the profile of Andrei Lyskov](https://cdn-images-1.medium.com/fit/c/72/72/1*xWjdj-z7-tuW2ZMU6lFW4g.jpeg)](https://towardsdatascience.com/@LyskovAndrei)

Andrei Lyskov 安德烈 · 利斯科夫





602







More from Towards Data Science 更多关于数据科学

The Next Level of Data Visualization in Python 数据可视化的下一个层次

[![Go to the profile of Will Koehrsen](https://cdn-images-1.medium.com/fit/c/72/72/1*SckxdIFfjlR-cWXkL5ya-g.jpeg)](https://towardsdatascience.com/@williamkoehrsen)

Will Koehrsen 威尔 · 科尔森





7K







More from Towards Data Science 更多关于数据科学

The Poisson Distribution and Poisson Process Explained 泊松分佈和泊松过程的解释

[![Go to the profile of Will Koehrsen](https://cdn-images-1.medium.com/fit/c/72/72/1*SckxdIFfjlR-cWXkL5ya-g.jpeg)](https://towardsdatascience.com/@williamkoehrsen)

Will Koehrsen 威尔 · 科尔森





1K





Responses 回应

![zhao shihuan](https://cdn-images-1.medium.com/fit/c/72/72/0*ChKB5CSx0kQrh_53)

Write a response… 写一个回复... ..

zhao shihuan 赵世桓



Conversation between 他们之间的对话[Arthur Paulino 阿瑟 · 保利诺](https://medium.com/@arthurpaulino) and 及[Will Koehrsen 威尔 · 科尔森](https://medium.com/@williamkoehrsen).

[![Go to the profile of Arthur Paulino](https://cdn-images-1.medium.com/fit/c/72/72/1*mnSzWEwKyMAJKKFGN5l_eA.jpeg)](https://medium.com/@arthurpaulino)

Arthur Paulino 阿瑟 · 保利诺

[Jun 25, 2018 2018年6月25日](https://medium.com/@arthurpaulino/keep-in-mind-that-these-ideas-should-be-validated-somehow-56062190b9e9?source=responses---------0---------------------)

keep in mind that these ideas should be validated somehow. i’d perform cross-validations to validate each insight separately.

记住，这些想法应该得到验证。 我会进行交叉验证，分别验证每个洞察力。



16

[1 response 1个响应](https://medium.com/@arthurpaulino/keep-in-mind-that-these-ideas-should-be-validated-somehow-56062190b9e9?source=responses---------0---------------------#--responses)

[![Go to the profile of Will Koehrsen](https://cdn-images-1.medium.com/fit/c/72/72/1*SckxdIFfjlR-cWXkL5ya-g.jpeg)](https://medium.com/@williamkoehrsen)

Will Koehrsen 威尔 · 科尔森

[Jun 26, 2018 2018年6月26日](https://medium.com/@williamkoehrsen/this-is-a-very-good-point-any-feature-selection-or-feature-engineering-you-do-to-a-dataset-da97d38527fa?source=responses---------0---------------------)

This is a very good point: any feature selection (or feature engineering) you do to a dataset should be thoroughly evaluated with cross validation to see if it has a beneficial effect. Machine learning is still largely an empirical field which means it’s nearly impossible to tell ahead of time what effect a particular choice will have on model…

这是一个非常好的观点: 对数据集所做的任何特征选择(或特征工程)都应该通过交叉验证进行彻底评估，看看它是否有好处。 机器学习在很大程度上仍然是一个经验领域，这意味着几乎不可能提前知道一个特定的选择会对模型产生什么影响..。

Read more… 阅读更多..



20





Conversation between 他们之间的对话[Rohit Wason 罗希特 · 瓦森](https://medium.com/@wasprobot) and 及[Will Koehrsen 威尔 · 科尔森](https://medium.com/@williamkoehrsen).

[![Go to the profile of Rohit Wason](https://cdn-images-1.medium.com/fit/c/72/72/0*VBDublFHf7mAgINZ.jpeg)](https://medium.com/@wasprobot)

Rohit Wason 罗希特 · 瓦森

[Jun 26, 2018](https://medium.com/@wasprobot/what-a-great-article-585e96156063?source=responses---------1---------------------)

What a great article! I am still going through the details of the functions in the FeatureSelector class, so it’ll take me more than the suggested 10-min to finish it, lol. But that only speaks to the simplicity you have brought to the act of mining a typical Kaggle-like dataset.

Kudos, man! I am following you.



5

[1 response](https://medium.com/@wasprobot/what-a-great-article-585e96156063?source=responses---------1---------------------#--responses)

[![Go to the profile of Will Koehrsen](https://cdn-images-1.medium.com/fit/c/72/72/1*SckxdIFfjlR-cWXkL5ya-g.jpeg)](https://medium.com/@williamkoehrsen)

Will Koehrsen

[Jun 29, 2018](https://medium.com/@williamkoehrsen/thanks-i-think-this-is-a-good-start-for-feature-selection-but-there-are-still-plenty-of-more-d01c0f8a1583?source=responses---------1---------------------)

Thanks! I think this is a good start for feature selection, but there are still plenty of more methods to implement. I’m looking forward to seeing how others can build on this and any contributions on GitHub are welcome!



4



Show all responses





- 

  5.5K 5.5 k

- 

- 

- 

![img](http://staging.caiyunapp.com/imgs/webtrs/fanyi-btn-hover.png)

![img](https://caiyunapp.com/imgs/webtrs/favorite-btn.png)