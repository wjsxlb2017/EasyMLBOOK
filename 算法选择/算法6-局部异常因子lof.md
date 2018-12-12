

# 算法原理-一句话介绍

局部异常因子（LOF）是一种基于密度的异常检测算法，通过比较样本点的局部密度（local density）和它邻居的局部密度来确定异常。


# 算法原理-文档
局部异常因子（LOF）是一种基于密度的异常检测算法。原理是比较样本点的局部密度（local density）和它邻近点的局部密度。如果两者差不多，则说明它是正常的，如果比邻近点的局部要明显稀疏，那就是异常点。

该算法最核心的概念就是局部密度，一个数据点的局部密度是由它周围的K个最近的邻居确定的，

# 算法原理-参数
- $$ k-distance(A)$$: 样本$$ {\displaystyle A}$$到它第K邻近的邻居的距离。注意在这个k邻居是指在这个距离内的所有样本点，真实个数可能不止K个，样本点集合定义为$$ {\displaystyle N_{k}(A)}$$.

- $$ reachability-distance_{k}$$: B到A的k-可达距离， 是A到B的真实距离, 并且至少是

$$ 
{\displaystyle {k-distance}(B)} 
$$,即：
$${\displaystyle {{reachability-distance}}_{k}(A,B)=\max({{k-distance}}(B),d(A,B))}$$
定义这种距离函数的目的是，对于B附近的K个邻居都赋予相同的距离，因为计算起密度来，这种方式比一般的欧式距离更加稳定。
- $$\displaystyle {{lrd}}(A)$$:  局部可达密度, 就是A的k邻居到A的可达距离的平均值，然后求个倒数。
$${\displaystyle {{lrd}}(A):=1/\left({\frac {\sum\limits _{B\in N_{k}(A)}{{reachability-distance}}_{k}(A,B)}{|N_{k}(A)|}}\right)}$$
- $${\displaystyle {{LOF}}_{k}(A)}$$: 样本点A的k邻居们的平局可达密度和A的局部可达密度的比值，
这个取值跟1越接近，说明A跟邻居的行为越像，这个值小于1时，说明A的相对邻居更稠密（inlier）, 这个值大于1时，说明A更像一个离群点，也就是异常值了。
- n_neighbors：n_neighbors表示邻居个数(一般取20).
- LOF得分：算法可以输出一个得分，这个得分越高，表示异常程度越大，得分越低表示很正常。所以在推断的时候，可以把这个参数当成敏感度。

