

------

![modin logo](https://modin.readthedocs.io/en/latest/_images/MODIN_ver2_hrz.png)

------
[TOC]
通过更改一行代码来扩展您的pandas工作流

要使用 Modin，请替换pandas导入:

![Fork me on GitHub](https://camo.githubusercontent.com/365986a132ccd6a44c23a9169022c0b5c890c387/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f7265645f6161303030302e706e67)

```python
# import pandas as pd
import modin.pandas as pd
```

# 通过更改一行代码来扩展您的pandas工作流

Modin 使用 Ray 提供了一种轻松的方式来加速您的pandas笔记本、脚本和库。 与其他分布式的 datafarme 库不同，Modin 提供了与现有 pandas 代码的无缝集成和兼容性。 即使使用 dataframeconstructor 也是相同的。

```python
import modin.pandas as pd
import numpy as np

frame_data = np.random.randint(0, 100, size=(2**10, 2**8))
df = pd.DataFrame(frame_data)
```

要使用 Modin，您不需要知道系统有多少个内核，也不需要指定如何分发数据。 事实上，您可以继续使用您以前的熊猫笔记本，同时经历一个相当大的加速从莫丁，甚至在一台机器。 一旦更改了导入声明，就可以像使用熊猫一样使用 Modin 了。

#  更快的pandas，甚至在你的笔记本电脑上

![Plot of read_csv](https://modin.readthedocs.io/en/latest/_images/read_csv_benchmark.png)



 `modin.pandas`的 DataFrame 是一个极其轻量级的并行 DataFrame。 Modin 透明地分发了数据和计算，因此您所需要做的就是继续使用 pandas API，就像安装 Modin 之前那样。 与其他并行 DataFrame 系统不同，Modin 是一个非常轻量级、健壮的 DataFrame。 由于它的重量很轻，Modin 在一台有4个物理核心的笔记本电脑上提供了高达4倍的速度提升。

在pandas中，当你进行任何形式的计算时，每次只能使用一个核。 使用 Modin时，你可以使用你机器上所有的 CPU 核。 即使在 read csv 中，我们也可以看到通过在整个机器上有效地分配工作而获得的巨大收益。

```python
import modin.pandas as pd

df = pd.read_csv("my_dataset.csv")
```

# 对从1KB 到1tb+的数据集来说，Modin 是一个 DataFrame

我们重点关注小数据(例如pandas)和大数据之间的 DataFrames 解决方案的桥接。 通常数据科学家需要不同的工具来对不同大小的数据做同样的事情。 现有的针对1KB 的 datatrame 解决方案不能扩展到1tb +的数据 ，而且对于1KB 范围的数据集来说，1tb + 解决方案的开销太大。 使用 Modin，由于其轻量级、可靠性和可伸缩性，您可以获得1KB 和1tb +的快速 DataFrame。

