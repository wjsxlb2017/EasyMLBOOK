





## grubb test

[Grubbs' Test](http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm)为一种假设检验的方法，常被用来检验服从正太分布的单变量数据集（univariate data set）YY 中的单个异常值。若有异常值，则其必为数据集中的最大值或最小值。原假设与备择假设如下：

H0H0: 数据集中没有异常值
H1H1: 数据集中有一个异常值

Grubbs' Test检验假设的所用到的检验统计量（test statistic）为

![image-20181102013109909](/Users/stellazhao/Library/Application Support/typora-user-images/image-20181102013109909.png)



其中，Y⎯⎯⎯⎯Y¯为均值，ss为标准差。原假设H0H0被拒绝，当检验统计量满足以下条件

![image-20181102013136428](/Users/stellazhao/Library/Application Support/typora-user-images/image-20181102013136428.png)



## sESD

在现实数据集中，异常值往往是多个而非单个。为了将Grubbs' Test扩展到kk个异常值检测，则需要在数据集中逐步删除与均值偏离最大的值（为最大值或最小值），同步更新对应的t分布临界值，检验原假设是否成立。基于此，Rosner提出了Grubbs' Test的泛化版[ESD](http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm)（Extreme Studentized Deviate test）。算法流程如下：

- 计算与均值偏离最远的残差，**注意**计算均值时的数据序列应是删除上一轮最大残差样本数据后；![image-20181102013242736](/Users/stellazhao/Library/Application Support/typora-user-images/image-20181102013242736.png)
- 计算临界值（critical value）；

![image-20181102013257845](/Users/stellazhao/Library/Application Support/typora-user-images/image-20181102013257845.png)

- 检验原假设，比较检验统计量与临界值；若Ri>λjRi>λj，则原假设H0H0不成立，该样本点为异常点；
- 重复以上步骤kk次至算法结束。