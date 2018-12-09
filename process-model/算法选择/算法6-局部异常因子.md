[TOC]


# 算法原理一句话介绍


局部异常因子（LOF）是一种基于密度的异常检测算法，通过比较样本点的局部密度（local density）和它邻居的局部密度来确定异常。


# 算法原理-文档

Local Outlier Factor（LOF）算法是一种基于密度的异常检典算法（Breuning et. al. 2000），算法原理是比较样本点的局部密度（local density）和它邻居的局部密度。如果两者差不多，则说明它是正常的，如果比邻居们的局部要明显稀疏，那就说明它是异常点。

该算法最核心的概念就是局部密度，一个数据点的局部密度是由它周围的K个最近的邻居确定的，

##  符号说明
- $$ k-distance(A)$$: 样本$$ {\displaystyle A}$$到它第K邻近的邻居的距离。注意在这个k邻居是指在这个距离内的所有样本点，真实个数可能不止K个，样本点集合定义为$$ {\displaystyle N_{k}(A)}$$.

- $$ reachability-distance_{k}$$: B到A的k-可达距离， 是A到B的真实距离, 并且至少是

$$ 
{\displaystyle {k-distance}(B)} 
$$,即：
$${\displaystyle {{reachability-distance}}_{k}(A,B)=\max({{k-distance}}(B),d(A,B))}$$
定义这种距离函数的目的是，对于B附近的K个邻居都赋予相同的距离，因为计算起密度来，这种方式比一般的欧式距离更加稳定（密度是距离的倒数呀，距离太小了，密度就无穷大了）。
- $$\displaystyle {{lrd}}(A)$$:  局部可达密度, 就是A的k邻居到A的可达距离的平均值，然后求个倒数。
$${\displaystyle {{lrd}}(A):=1/\left({\frac {\sum\limits _{B\in N_{k}(A)}{{reachability-distance}}_{k}(A,B)}{|N_{k}(A)|}}\right)}$$
- ${\displaystyle {{LOF}}_{k}(A)}$: 样本点A的k邻居们的平局可达密度和A的局部可达密度的比值，
这个取值跟1越接近，说明A跟邻居的行为越像，这个值小于1时，说明A的相对邻居更稠密（inlier）, 这个值大于1时，说明A更像一个离群点，也就是异常值了。

## 延伸
LOF算法中关于局部可达密度的定义其实暗含了一个假设，即：不存在大于等于k个重复的点。当这样的重复点存在的时候，这些点的平均可达距离为零，局部可达密度就变为无穷大，会给计算带来一些麻烦。在实际应用时，为了避免这样的情况出现，可以把 k-distance 改为 k-distinct-distance，不考虑重复的情况。或者，还可以考虑给可达距离都加一个很小的扰动项，避免可达距离等于零。
LOF算法需要计算数据点两两之间的距离，造成整个算法时间复杂度为 $O(n^2)$ 。为了提高算法效率，后续有算法尝试改进。FastLOF （Goldstein，2012）先将整个数据随机的分成多个子集，然后在每个子集里计算 LOF 值。对于那些 LOF 异常得分小于等于 1 的（inliers），从数据集里剔除，剩下的在下一轮寻找更合适的 nearest-neighbor，并更新 LOF 值。这种先将数据粗略分成多个部分，然后根据局部计算结果将数据过滤来减少计算量的想法，并不罕见。比如，为了改进 K-means 的计算效率， Canopy Clustering 算法也采用过比较相似的做法。


# 训练参数简化
LOF计算距离矩阵的过程对应为算法的训练，最关键的参数就是邻居的个数K(sklearn里面的n_neighbors， 一般取20) 
- K越大, $N_{k}(A)$集合就越大，更多的训练集会被划入inlier，
- K越小, $N_{k}(A)$集合就越小，更多的训练集会被划入outlier，

# 应用参数简化
LOF算法本身可以输出一个得分，这个得分越高，表示异常程度越大，得分越低表示很正常。
所以在推断的时候，可以把这个参数当成敏感度。

# 参考文献

1. M. M. Breunig, H. P. Kriegel, R. T. Ng, J. Sander. LOF: Identifying Density-based Local Outliers. SIGMOD, 2000.
2. M. Goldstein. FastLOF: An Expectation-Maximization based Local Outlier detection algorithm. ICPR, 2012
3. https://en.wikipedia.org/wiki/Local_outlier_factor
