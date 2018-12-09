[TOC]

## RNN的输入和输出代表什么意思

对于堆叠多层lstm，lstm的超参和输出的数据的多个维度代表什么意思。

如下是堆叠多层lstm在时间这个维度上的一个截面

   ![](/Users/stellazhao/tencent_workplace/gitlab/dataming/EasyML/doc/algorithm/_image/2018-09-24-15-30-29.jpg)

如下的多层RNN，总共有三层（输入，隐藏层，输出）, 如下的时间窗口为4，这4个网络的参数是共享的。单独来看每个时刻的切面，就是一个MLP的增强版：原始的MLP的基础上，上一个时刻的hidden cell的值，需要传给本次timestep对应MLP的hidden cell作为输入。

![](/Users/stellazhao/tencent_workplace/gitlab/dataming/EasyML/doc/algorithm/_image/2018-09-24-15-32-48.jpg)

上面讲了rnn的运行机制，至于更复杂的rnn比如LSTM、RGU，那就是在MLP的每个hidden cell（一个黄色circle）与下一个time_step 的hidden cell的传值机制的more sophisticated tactics。



## RNN的常用结果举例

- 1对1：时间步长为1，退化为MLP, 比如图像分类。
- 1对多：图片取标题。
- 多对1：情感分析/时序分类/时序异常检测。
- 多对多1：机器翻译/多步骤时序预测。
- 多对多2：视频分类，实时的对每一帧画面标记。

注意rnn模型对输入序列的长度没有限制，只要是单个时间长度的整数倍就行了，因为中间的隐藏层是可以使用任意多次。如下是在时间轴上的切片，每个格子代表的是多个cell。

![image-20181023034957036](/Users/stellazhao/tencent_workplace/gitlab/dataming/EasyML/doc/algorithm/_image/rnn_deploy.png)

## 参考资料

1. https://www.zhihu.com/question/41949741?sort=created
2. https://blog.csdn.net/jiangpeng59/article/details/77646186
3. https://github.com/Vict0rSch/deep_learning