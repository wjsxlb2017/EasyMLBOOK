
[TOC]

# 算法原理-句话说明

DBSCAN算法用作异常检测的原理是，在聚类过程中通过寻找核心点来扩展类簇边界，得到样本空间中不同的高密度区域，而没有落入高密度区域的样本点就被视为异常点。



# 算法原理-文档



DBSCAN算法用作异常检测的原理是，在聚类过程中通过寻找核心点来扩展类簇边界，得到样本空间中不同的高密度区域，而没有落入高密度区域的样本点就被视为异常点。



具体来说,DBSCAN算法将样本点定义为三种类型：

- ![](./_image/2018-09-15-23-13-34.png)

- 核心点（core points)：下图中的红色点，满足$\epsilon $邻域内的邻居个数超过minPoints。
- 边缘点（border points）：类簇边缘的点。
- 异常点(outliers): 如果某个点附近不存在可达点，那就定义为异常点，下图中的蓝色点。

另一个类簇中的core points 和border points之间存在两种空间关系：
  - 直接可达(directly reachable points):如果p是核心点，那么p的$\epsilon $邻域内所有的点都是直接可达点。
    如果p 是core points且$distance(p, q) < \epsilon$, 就说q能直接可达p。
  - 可达（reachable points）:如果p是核心点，p直接可达另外的核心点p2，p10，而p10直接可达q，则说q可达p。
    如果存在路径$p_1, \dots, p_n $且 $p_1 = p, p_n = q$, 这里$p_{i+1}$直接可达$p_{i}$，路径上除了$p_n$都必须为core points , 就说q可达p。下图中B到A是可达的，C到A也是可达的。
  - 密度可连（density-connected ）：上面定义的可达是一种非对称的关系，对于两个none-core的点p，q，如何定义“可达”这种位置关系呢？这里引入了密度可达的概念，如果存在core point-o，使得p 和 q都可达o，则p和q密度可连。
  - 类簇： 基于密度可连的定义，一个类簇里面所有的点都是密度可连的。如果某个点可对簇里面的任意一个点都是密度可达的，那么这个点也属于这个簇。


聚类的训练 ，即构造类簇的过程如下：
- 查找每个数据点的 $\epsilon-$ 邻居，并且标记core points。.
- 在图中查找连接起来的core points，忽略掉non-core points。
- 如果non-core point最近的类簇是$\epsilon-邻居$，就把non-core point分配给该类簇，否则它就是噪声。

## 参数说明

训练过程最关键的参数：
- 1. epsilon: 邻域的大小。
- 2. minPoints: core-point附近邻居的个数.
     其他参数：
     
###  [附]调参技巧之如何选择合适的epsilon[^3]

关于超参epsilon，即半径大小该如何设置呢？
通过计算K-nn距离，将距离从小到大排序，得到一条单调曲线，然后寻找曲线的拐点（elbow or knee of a curve）， 关于如何寻找拐点可以参考[^1].大概原理(右下图)是，在曲线上寻找点p, 使得p离曲线两个端点连线的距离最大。
![企业微信截图_ad1b2831-6ff2-40d1-acc7-40a36435e2af](/Users/stellazhao/statistics_studyplace/EasyML/doc/algorithm/_image/choose_epsilon_for_dbscan.png)

下面再举个例子说明
![企业微信截图_30f968ae-3bcf-4175-8acb-aa9f6521c35d](/Users/stellazhao/statistics_studyplace/EasyML/doc/algorithm/_image/find_knee.png)

# 算法原理-可视化demo

在选择算法的环节，为了让用户能快速了解算法和上手操作，平台提文档以及可跟用户交互的形式，对算法的原理和超参的含义解释说明，演示算法的内部运行机制，。



demo中需要增加文字说明：

下图是DBSCAN算法可视化的例子(由meifan提供),  通过交互图展示训练过程可视化和训练结果。

两个关键参数
​     - 1. epsilon: 邻域的大小
- 2. minPoints: core-point附近邻居的个数.
  如下图所示   
  ![dbscan_参数解释](./_image/dbscan_参数解释.png)
  点击"run"就开始进入训练过程可视化。
  ![dbscan_training](./_image/dbscan_training.png)


## 【附】额外的开发工作

1. DBSCAN是一个offline检测的算法，如何应用到online检测，我们需要做如下改进和优化：
将算法拆分成训练和应用两部分。
	1. 训练过程：使用样本数据构造core-points，以及cluster。
	2. 应用过程:  对于测试数据，查找它的$\epsilon$ points。
2. DBSCAN是一个基于聚类的算法，他只能输出0/1, 即是否正常，如果要得到异常得分，我们需要做一些额外的工作。


# 场景可视化-(交互式demo)

针对时间序列异常检测场景，该算法可以输出每个时刻的异常程度以及是否异常的结果。
如下图所示，横轴表示时间，蓝色的周期性波动曲线表示检测的指标，
橙色的曲线表示算法输出的该时刻曲线取值的异常程度
黑色的点表示算法检测出来的异常点，可以看到，当橙色的曲线上的点即异常分值超过了某个阈值之后，
该时刻的点就被判断为异常。
![image-20181120134644929](/Users/stellazhao/statistics_studyplace/EasyML/doc/algorithm/_image/场景化-可视化.png)





#  [附]应用参数抽象

predict方法用来预测新样本点是否属于已有的类簇。属于某个类的定义是：如果在某类中存在样本点，使得新的样本点位于该样本点的$\epsilon$邻域内，那么新样本点就属于这个类簇。不属于任何一类的样本点，就是噪声点或者异常点.

![image-20181110160302686](/Users/stellazhao/Library/Application Support/typora-user-images/image-20181110160302686.png)

predict逻辑如下

newdata是即将

```R
function (object, newdata = NULL, data, ...)
{
    if (is.null(newdata))
        return(object$cluster)
    # 计算测试数据跟训练样本中的knn矩阵，注意这里需要排序
    nn <- frNN(rbind(data, newdata), eps = object$eps, sort = TRUE,
        ...)$id[-(1:nrow(data))]
    sapply(nn, function(x) {
        # 计算的邻居节点中，只保留训练样本中的节点。x是训练样本点的index
        x <- x[x <= nrow(data)]
        # object$cluster[x]：训练样本点对应的label
        # object$cluster[x][x > 0]：将样本点中的noise节点过滤掉
        # object$cluster[x][x > 0][1]：取剩下的符合条件的训练样本中最近的一个样本的label
        x <- object$cluster[x][x > 0][1]
        # 如果通过上面的筛选，一个符合条件的邻居都没有，那么就是异常点了
        x[is.na(x)] <- 0L
        x
    })
}
```

frNN

```R
function (x, eps, sort = TRUE, search = "kdtree", bucketSize = 10,
    splitRule = "suggest", approx = 0)
{
    if (is(x, "frNN")) {
        if (x$eps < eps)
            stop("frNN in x has not a sufficient eps radius.")
        for (i in 1:length(x$dist)) {
            take <- x$dist[[i]] <= eps
            x$dist[[i]] <- x$dist[[i]][take]
            x$id[[i]] <- x$id[[i]][take]
        }
        x$eps <- eps
        return(x)
    }
    search <- pmatch(toupper(search), c("KDTREE", "LINEAR", "DIST"))
    if (is.na(search))
        stop("Unknown NN search type!")
    if (search == 3) {
        if (!is(x, "dist"))
            if (.matrixlike(x))
                x <- dist(x)
            else stop("x needs to be a matrix to calculate distances")
    }
    if (is(x, "dist")) {
        if (any(is.na(x)))
            stop("data/distances cannot contain NAs for frNN (with kd-tree)!")
        x <- as.matrix(x)
        diag(x) <- Inf
        id <- lapply(1:nrow(x), FUN = function(i) {
            y <- x[i, ]
            o <- order(y, decreasing = FALSE)
            o[y[o] <= eps]
        })
        names(id) <- rownames(x)
        d <- lapply(1:nrow(x), FUN = function(i) {
            unname(x[i, id[[i]]])
        })
        names(d) <- rownames(x)
        return(structure(list(dist = d, id = id, eps = eps, sort = TRUE),
            class = c("frNN", "NN")))
    }
    if (!.matrixlike(x))
        stop("x needs to be a matrix to calculate distances")
    x <- as.matrix(x)
    if (storage.mode(x) == "integer")
        storage.mode(x) <- "double"
    if (storage.mode(x) != "double")
        stop("x has to be a numeric matrix.")
    splitRule <- pmatch(toupper(splitRule), .ANNsplitRule) -
        1L
    if (is.na(splitRule))
        stop("Unknown splitRule!")
    if (any(is.na(x)))
        stop("data/distances cannot contain NAs for frNN (with kd-tree)!")
    ret <- frNN_int(as.matrix(x), as.double(eps), as.integer(search),
        as.integer(bucketSize), as.integer(splitRule), as.double(approx))
    if (sort) {
        o <- lapply(1:length(ret$dist), FUN = function(i) order(ret$dist[[i]],
            ret$id[[i]], decreasing = FALSE))
        ret$dist <- lapply(1:length(o), FUN = function(p) ret$dist[[p]][o[[p]]])
        ret$id <- lapply(1:length(o), FUN = function(p) ret$id[[p]][o[[p]]])
    }
    names(ret$dist) <- rownames(x)
    names(ret$id) <- rownames(x)
    ret$eps <- eps
    ret$sort <- sort
    class(ret) <- c("frNN", "NN")
    ret
}
```








# r程序示例
```R
 data(iris)
 iris <- as.matrix(iris[,1:4])

 ## find suitable eps parameter using a k-NN plot for k = dim + 1
 ## Look for the knee!
 kNNdistplot(iris, k = 5)
 abline(h=.5, col = "red", lty=2)

 res <- dbscan(iris, eps = .5, minPts = 5)
 res

 pairs(iris, col = res$cluster + 1L)

 ## use precomputed frNN
 fr <- frNN(iris, eps = .5)
 dbscan(fr, minPts = 5)

 ## example data from fpc
 set.seed(665544)
 n <- 100
 x <- cbind(
   x = runif(10, 0, 10) + rnorm(n, sd = 0.2),
   y = runif(10, 0, 10) + rnorm(n, sd = 0.2)
   )

 res <- dbscan(x, eps = .3, minPts = 3)
 res

 ## plot clusters and add noise (cluster 0) as crosses.
 plot(x, col=res$cluster)
 points(x[res$cluster==0,], pch = 3, col = "grey")

 hullplot(x, res)

 ## predict cluster membership for new data points
 ## (Note: 0 means it is predicted as noise)
 newdata <- x[1:5,] + rnorm(10, 0, .2)
 predict(res, x, newdata)

 ## compare speed against fpc version (if microbenchmark is installed)
 ## Note: we use dbscan::dbscan to make sure that we do now run the
 ## implementation in fpc.
 ## Not run:

 if (requireNamespace("fpc", quietly = TRUE) &&
     requireNamespace("microbenchmark", quietly = TRUE)) {
   t_dbscan <- microbenchmark::microbenchmark(
     dbscan::dbscan(x, .3, 3), times = 10, unit = "ms")
   t_dbscan_linear <- microbenchmark::microbenchmark(
     dbscan::dbscan(x, .3, 3, search = "linear"), times = 10, unit = "ms")
   t_dbscan_dist <- microbenchmark::microbenchmark(
     dbscan::dbscan(x, .3, 3, search = "dist"), times = 10, unit = "ms")
   t_fpc <- microbenchmark::microbenchmark(
     fpc::dbscan(x, .3, 3), times = 10, unit = "ms")

   r <- rbind(t_fpc, t_dbscan_dist, t_dbscan_linear, t_dbscan)
   r

   boxplot(r,
     names = c('fpc', 'dbscan (dist)', 'dbscan (linear)', 'dbscan (kdtree)'),
     main = "Runtime comparison in ms")

   ## speedup of the kd-tree-based version compared to the fpc implementation
   median(t_fpc$time) / median(t_dbscan$time)
 }
 ## End(Not run)
```



# 参考文献

[1]. https://en.wikipedia.org/wiki/DBSCAN

[2]. https://cran.r-project.org/package=dbscan/dbscan.pdf

[3] Martin Ester, Hans-Peter Kriegel, Joerg Sander, Xiaowei Xu (1996).
A Density-Based Algorithm for Discovering Clusters in Large
Spatial Databases with Noise. Institute for Computer Science,
University of Munich. _Proceedings of 2nd International Conference
on Knowledge Discovery and Data Mining (KDD-96)._

[4] Campello, R. J. G. B.; Moulavi, D.; Sander, J. (2013).
Density-Based Clustering Based on Hierarchical Density Estimates.
_Proceedings of the 17th Pacific-Asia Conference on Knowledge
Discovery in Databases, PAKDD 2013,_ Lecture Notes in Computer
Science 7819, p. 160.

[^1]: <https://www1.icsi.berkeley.edu/~barath/papers/kneedle-simplex11.pdf>
[^3 ]: <https://www.researchgate.net/post/How_can_I_choose_eps_and_minPts_two_parameters_for_DBSCAN_algorithm_for_efficient_results>

