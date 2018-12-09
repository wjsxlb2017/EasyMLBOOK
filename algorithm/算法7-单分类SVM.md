[TOC]


# 算法原理
单分类SVM的原理是将所有的样本与零点在特征空间F中分离开，并且分的越开越好。
具体来说
训练过程（training）：构建特征空间的概率密度函数并估计参数，模型假设所有训练样本都是位于概率密度的中心处，除了原点。
推断过程（serving）：
对于新来的样本点，使用训练得到的概率密度函数计算它的概率，低于一定阈值返回1，即异常点，否则返回正常。

## 1. OCSVM or $\nu-SVM$
这个方法创建了一个参数为$(\omega,\rho)$的超平面，该超平面与特征空间$F$中的零点距离最大，并且将零点与所有的数据点分隔开, 可以表述为优化问题:

$\min\limits_{\omega,\zeta_{i},\rho}\frac{1}{2}||w||^{2}+\frac{1}{\nu n}\sum\limits_{i=1}^{n}{\zeta_{i}}-\rho ,\\ \tag{1} s.t. \omega^{T}\phi(x_{i})>\rho-\zeta_{i} , i=1,...,n\\ \zeta_{i}>0 ,i=1,...,n $

符号说明:
- 超参:
  - $\nu $ 类似二分类SVM中的C, 它是间隔误差（margin error）的上界, 
  是训练集中做为支持向量的样例占比的下界。
  e.g., 如果$\nu = 0.05$, 至多有5%的样本被错误分类（在当前训练出来的决策超平面），
  至少有5%的训练实例被当成了（当前训练出来的决策超平面）支撑向量。
  
- 待估参数
  - $\phi$表示在一个映射函数，将特征从原始的空间映射到新的特征空间$F$.
  - $\zeta_{i}$ 是松弛变量。
  - $\rho$ 表示超平面距离原点的距离。
  - $\omega$ 表示超平面的法向量。

因为这个参数的重要性，这种方法也被称为 $\nu-SVM$ 。采用Lagrange技术并且采用dot-product calculation，预测某个样本是否为异常的函数变为：

$f(x)=sgn(\omega ^{T}\phi(x_{i})-\rho) =sgn(\sum\limits_{i=1}^{n}{\alpha_{i}K(x,x_{i})}-\rho) $

## 2. SVDD
另外一种算法-The method of Support Vector Data Description by Tax and Duin (SVDD)采用一个超球面而不是超平面的方法，该算法在特征空间中获得数据周围的球形边界，当超球体的球面面积最小时，对应的超球面就是决策边界，边界外的样本是异常，边界内是正常。
产生的超球体参数为$(a, R)$,a是球体中心,它是支持向量的线性组合； R是球体半径,中心$a$跟传统SVM方法相似，可以要求所有数据 $x_{i}$到中心的距离严格小于R，但是更加robust的做法是通过构造一个惩罚系数为C的松弛变量 $\zeta_{i}，i=1,...,n$ ，求解如下优化问题：

$\min\limits_{R,a, \zeta_{i}}R^{2}+C\sum\limits_{i=1}^{n}{\zeta_{i}} \\ s.t. ||x_{i}-a||^{2}\leq R^{2}+\zeta_{i},i=1,...,n \\ \zeta_{i}\geq 0,i=1,...,n $
符号说明:
- 超参
  - C: 惩罚系数
- 待估参数
  - $R$: 球体半径
  - $a$: 球体中心
  - $\zeta_{i}$: 松弛变量
- $x_i$: 第i个样本的特征向量.

距离函数采用Gaussian Kernel：
$||z-x||^{2}=\sum\limits_{i=1}^{n}{a_{i}\exp(\frac{-||z-x_{i}||^{2}}{\sigma^{2}})}\geq-R^{2}/2+C_{R} $
如果z到中心a的距离小于或者等于半径， 判断新的数据点 z 是否在类内，。

# 训练参数简化
## 1. OCSVM or $\nu$-SVM

  对于OCSVM, 关键的参数为$\nu$, $\nu $ 类似二分类SVM中的C, 它是间隔误差（margin error）的上界, 
  是训练集中做为支持向量的样例占比的下界。$\nu = A + B/C$
  e.g., 如果$\nu = 0.05$, 至多有5%的样本被错误分类（在当前训练出来的决策超平面），
  至少有5%的训练实例被当成了（当前训练出来的决策超平面）支撑向量。
  由于待估计的超平面一侧是原点 + 少数越界的点，另一侧是大部分正常的点。
  - 当$\nu$取值越小，对松弛变量的惩罚就越大，这就会使得超平面把更多的训练样本划分到了正常的一测.
  - 当$\nu$取值越大，对松弛变量的惩罚就越小, 这就会使得超平面把更多的训练样本划分到了异常的一测.


## 2. SVDD
- 训练
  对于SVDD算法，关键的参数为C,C越大对“越界”的训练样本的惩罚就越大，参数估计的超球体半径就越大，从而对“异常”的定义就更苛刻，导致训练样本中有更少的样本被划分成于“异常”。考虑一种极端情况，当C取无穷大时，上面的优化问题退化为，求解一个半径最小的超球体，使得每个样本点都落在超球体中。
  对应到异常检测场景中：C越大，训练出的模型对异常的容忍度越高，检测算法更不敏感。

# 应用参数抽象

## 1. OCSVM or $\nu$-SVM
  默认的判别函数
  $f(x)=sgn(\omega ^{T}\phi(x_{i})-\rho) =sgn(\sum\limits_{i=1}^{n}{\alpha_{i}K(x,x_{i})}-\rho) $
  模型的使用者可以通过修改判断的阈值$\alpha$（类似N sigma算法的N），来调成模型的敏感度，具体操作为:
  $f(x)=sgn(\omega ^{T}\phi(x_{i})-\rho) =sgn(\sum\limits_{i=1}^{n}{\alpha_{i}K(x,x_{i})}- \alpha * \rho) $


## 2. SVDD
  训练阶段，得到了超球体的中心和半径后，在预测阶段，模型会输出的是一个离球心的距离和半径，默认的判断准则是：
  - 如果$||z-a||^{2} > R^2$, z是异常
  - 如果$||z-a||^{2} <= R^2$, z是正常

  模型的使用者可以通过修改判断的阈值$\alpha$（类似N sigma算法的N），来调成模型的敏感度，具体操作为:
​    - 如果$||z-a||^{2} > \alpha * R^2$, z是异常
​    - 如果$||z-a||^{2} <=  \alpha * R^2$, z是正常.

# 参考资料
[1] The Support Vector Method For Novelty Detection by Schölkopf et al.
[2] The method of Support Vector Data Description by Tax and Duin (SVDD)
[3] https://zhuanlan.zhihu.com/p/32784067
[4] https://stackoverflow.com/questions/11230955/what-is-the-meaning-of-the-nu-parameter-in-scikit-learns-svm-class
