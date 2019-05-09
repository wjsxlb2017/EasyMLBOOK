#  Creating Bitcoin trading bots don’t lose money

# 创建不会赔钱c比特币交易机器人

## Let’s make cryptocurrency-trading agents using deep reinforcement learning

## 让我们使用深度强化学习进行加密货币交易代理





[Adam King 亚当 · 金](https://towardsdatascience.com/@notadamking)Follow 跟着

Apr 28 4月28日

In this article we are going to create deep reinforcement learning agents that learn to make money trading Bitcoin. In this tutorial we will be using OpenAI’s `gym` and the PPO agent from the `stable-baselines` library, a fork of OpenAI’s `baselines` library.

在这篇文章中，我们将要创建一个深入的比特币强化学习代理商，学习如何通过比特币交易赚钱。 在本教程中，我们将使用 OpenAI 的健身房和 PPO 代理从稳定的基线库，一个分支的 OpenAI 的基线库。

Many thanks to OpenAI and DeepMind for the open source software they have been providing to deep learning researchers for the past couple of years. If you haven’t yet seen the amazing feats they’ve accomplished with technologies like [AlphaGo, OpenAI Five, and AlphaStar](https://openai.com/blog/), you may have been living under a rock for the last year, but you should also go check them out.

非常感谢 OpenAI 和 DeepMind 为深度学习研究人员提供的开源软件。 如果你还没有见识过 AlphaGo、 OpenAI Five 和 AlphaStar 等技术所取得的惊人成就，你可能去年一直生活在岩石下，但你也应该去看看它们。



![img](https://cdn-images-1.medium.com/max/1600/0*IeiYxZVLPlPmbG38.png)

AlphaStar Training ( Alphastar 培训课程(<https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/>)

While we won’t be creating anything quite as impressive, it is still no easy feat to trade Bitcoin profitably on a day-to-day basis. However, as Teddy Roosevelt once said,

虽然我们不会创造出如此令人印象深刻的东西，但是在日常基础上以盈利的方式交易比特币仍然不是一件容易的事情。 然而，正如泰迪 · 罗斯福曾经说过的,

> Nothing worth having comes easy. 值得拥有的东西来之不易

So instead of learning to trade ourselves… let’s make a robot to do it for us.

所以，与其学习交易我们自己... ... 不如让我们制造一个机器人来为我们做这件事。

### The Plan

### 计划



![img](https://cdn-images-1.medium.com/max/1600/1*r7XItmcyWv76mso08vncpw.jpeg)

1. Create a gym environment for our agent to learn from 为我们的代理人创造一个可以学习的健身环境
2. Render a simple, yet elegant visualization of that environment 呈现该环境的简单但优雅的可视化效果
3. Train our agent to learn a profitable trading strategy 训练我们的代理商学习有利可图的交易策略

If you are not already familiar with [how to create a gym environment from scratch](https://medium.com/@notadamking/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e), or [how to render simple visualizations of those environments](https://medium.com/@notadamking/visualizing-stock-trading-agents-using-matplotlib-and-gym-584c992bc6d4), I have just written articles on both of those topics. Feel free to pause here and read either of those before continuing.

如果你还不熟悉如何从零开始创建一个健身房环境，或者如何呈现这些环境的简单可视化效果，我刚刚就这两个主题写了一些文章。 在继续之前，你可以在这里暂停并阅读其中的任何一个。

------

### Getting Started

### 开始

For this tutorial, we are going to be using the Kaggle data set produced by [Zielak](https://www.kaggle.com/mczielinski/bitcoin-historical-data). The `.csv` data file will also be available on my [Github](https://github.com/notadamking/Bitcoin-Trader-RL) repo if you’d like to download the code to follow along. Okay, let’s get started.

在本教程中，我们将使用 Zielak 生成的 Kaggle 数据集。 这个。 如果你想下载后续代码，csv 数据文件也可以在我的 Github repo 上找到。 好了，我们开始吧。

First, let’s import all of the necessary libraries. Make sure to `pip install` any libraries you are missing.

首先，让我们导入所有必要的库。 确保安装缺少的库时弹出 pip 命令。

```
import gym
import pandas as pd
import numpy as np
from gym import spaces
from sklearn import preprocessing
```

Next, let’s create our class for the environment. We’ll require a `pandas` data frame to be passed in, as well as an optional `initial_balance`, and a `lookback_window_size`, which will indicate how many time steps in the past the agent will observe at each step. We will default the `commission` per trade to 0.075%, which is Bitmex’s current rate, and default the `serial` parameter to false, meaning our data frame will be traversed in random slices by default.

接下来，让我们为环境创建类。 我们将需要传入一个熊猫数据帧，以及一个可选的初始平衡，和一个回溯窗口大小，它将显示代理将在每个步骤中观察过去的时间步长。 我们将每笔交易的佣金默认为0.075% ，这是 Bitmex 的当前利率，并将序列参数默认为 false，这意味着我们的数据框架将默认为随机切片。

We also call `dropna()` and `reset_index()` on the data frame to first remove any rows with `NaN` values, and then reset the frame’s index since we’ve removed data.

我们还在数据帧上调用 dropna ()和 reset index () ，以首先删除任何带有 NaN 值的行，然后在删除数据后重置帧的索引。

```
class BitcoinTradingEnv(gym.Env):
  """A Bitcoin trading environment for OpenAI gym"""
  metadata = {'render.modes': ['live', 'file', 'none']}
  scaler = preprocessing.MinMaxScaler()
  viewer = None
  def __init__(self, df, lookback_window_size=50, 
                         commission=0.00075,  
                         initial_balance=10000
                         serial=False):
    super(BitcoinTradingEnv, self).__init__()
  self.df = df.dropna().reset_index()
    self.lookback_window_size = lookback_window_size
    self.initial_balance = initial_balance
    self.commission = commission
    self.serial = serial
  # Actions of the format Buy 1/10, Sell 3/10, Hold, etc.
    self.action_space = spaces.MultiDiscrete([3, 10])
  # Observes the OHCLV values, net worth, and trade history
    self.observation_space = spaces.Box(low=0, high=1, shape=(10, 
                    lookback_window_size + 1), dtype=np.float16)
```

Our `action_space` here is represented as a discrete set of 3 options (buy, sell, or hold) and another discrete set of 10 amounts (1/10, 2/10, 3/10, etc). When the buy action is selected, we will buy `amount * self.balance` worth of BTC. For the sell action, we will sell `amount * self.btc_held` worth of BTC. Of course, the hold action will ignore the amount and do nothing.

我们这里的操作空间表示为一个由3个选项(买入、卖出或持有)和另一个由10个金额(1 / 10、2 / 10、3 / 10等)组成的离散集合。 当购买行为被选择，我们将购买金额 * 自己。余额价值 BTC。 对于卖出行为，我们将卖出价值 * 的 BTC。 当然，持有操作会忽略金额，什么也不做。

Our `observation_space` is defined as a continuous set of floats between 0 and 1, with the shape `(10, lookback_window_size + 1)`. The `+ 1` is to account for the current time step. For each time step in the window, we will observe the OHCLV values, our net worth, the amount of BTC bought or sold, and the total amount in USD we’ve spent on or received from those BTC.

我们的观察空间定义为一组连续的浮点数，其形状介于0和1之间(10，回望窗口大小 + 1)。 + 1表示当前时间步长。 对于窗口中的每一个时间步骤，我们将观察 OHCLV 的价值，我们的净资产，BTC 的买入或卖出金额，以及我们在这些 BTC 上花费或收到的美元总额。

Next, we need to write our `reset` method to initialize the environment.

接下来，我们需要编写 reset 方法来初始化环境。

```
def reset(self):
  self.balance = self.initial_balance
  self.net_worth = self.initial_balance
  self.btc_held = 0
  self._reset_session()
  self.account_history = np.repeat([
    [self.net_worth],
    [0],
    [0],
    [0],
    [0]
  ], self.lookback_window_size + 1, axis=1)
  self.trades = []
  return self._next_observation()
```

Here we use both `self._reset_session` and `self._next_observation`, which we haven’t defined yet. Let’s define them.

在这里我们同时使用两个自我。 重置会话和自我。 接下来的观察，我们还没有定义。 让我们来定义它们。

#### Trading Sessions

#### 交易日



![img](https://cdn-images-1.medium.com/max/1600/1*hor57pXvQR42QmW-mIS5ew.jpeg)

An important piece of our environment is the concept of a trading session. If we were to deploy this agent into the wild, we would likely never run it for more than a couple months at a time. For this reason, we are going to limit the amount of continuous frames in `self.df` that our agent will see in a row.

我们的环境的一个重要部分是交易时段的概念。 如果我们将这个代理部署到外部环境中，我们可能一次运行它不会超过两个月。 由于这个原因，我们将限制我们的代理在一行中看到的连续帧的数量。

In our `_reset_session` method, we are going to first reset the `current_step`to `0`. Next, we’ll set `steps_left` to a random number between `1` and `MAX_TRADING_SESSION`, which we will now define at the top of the file.

在我们的 reset session 方法中，我们将首先将当前步骤重置为0。 接下来，我们将左边的步骤设置为1和 MAX trading session 之间的一个随机数字，现在我们将在文件顶部定义这个数字。

```
MAX_TRADING_SESSION = 100000  # ~2 months
```

Next, if we are traversing the frame serially, we will setup the entire frame to be traversed, otherwise we’ll set the `frame_start` to a random spot within `self.df`, and create a new data frame called `active_df`, which is just a slice of `self.df` from `frame_start` to `frame_start + steps_left`.

接下来，如果我们连续遍历框架，我们将设置要遍历的整个框架，否则我们将把框架 start 设置为 self.df 中的一个随机点，并创建一个名为 active df 的新数据框架，它只是一个 self.df 片段，从框架开始到框架开始 + 步骤左。

```
def _reset_session(self):
  self.current_step = 0
  if self.serial:
    self.steps_left = len(self.df) - self.lookback_window_size - 1
    self.frame_start = self.lookback_window_size
  else:
    self.steps_left = np.random.randint(1, MAX_TRADING_SESSION)
    self.frame_start = np.random.randint(
         self.lookback_window_size, len(self.df) - self.steps_left)
  self.active_df = self.df[self.frame_start -   
       self.lookback_window_size:self.frame_start + self.steps_left]
```

One important side effect of traversing the data frame in random slices is our agent will have much more *unique* data to work with when trained for long periods of time. For example, if we only ever traversed the data frame in a serial fashion (i.e. in order from `0` to `len(df)`), then we would only ever have as many unique data points as are in our data frame. Our observation space could only even take on a discrete number of states at each time step.

在随机切片中遍历数据帧的一个重要副作用是，经过长时间的训练，我们的代理将有更多独特的数据可以处理。 例如，如果我们只是以串行方式遍历数据帧(即从0到 len (df)) ，那么我们将只有数据帧中的那么多唯一数据点。 我们的观测空间在每个时间步骤中甚至只能呈现一个离散的状态数。

However, by randomly traversing slices of the data frame, we essentially manufacture more unique data points by creating more interesting combinations of account balance, trades taken, and previously seen price action for each time step in our initial data set. Let me explain with an example.

然而，通过随机遍历数据框架的切片，我们实质上通过创建更多有趣的账户余额、交易和之前看到的初始数据集中每个时间步的价格动作的组合来制造更多独特的数据点。 让我用一个例子来解释。

At time step 10 after resetting a serial environment, our agent will always be at the same time within the data frame, and would have had 3 choices to make at each time step: buy, sell, or hold. And for each of these 3 choices, another choice would then be required: 10%, 20%, …, or 100% of the amount possible. This means our agent could experience any of (1⁰³)¹⁰ total states, for a total of 1⁰³⁰ possible unique experiences.

在重置序列环境后的第10个步骤中，我们的代理将始终处于数据框架内的同一时间，并且在每个时间步骤中有3个选择: 买入、卖出或持有。 对于这三个选择中的每一个，另一个选择是必需的: 10% ，20% ，... ，或者100% 的可能数量。 这意味着我们的代理可以体验任何(103)10个国家，总共1030个可能的独特经验。

Now consider our randomly sliced environment. At time step 10, our agent could be at any of `len(df)` time steps within the data frame. Given the same choices to make at each time step, this means this agent could experience any of `len(df)`³⁰ possible unique states within the same 10 time steps.

现在考虑一下我们随机分割的环境。 在时间步骤10中，我们的代理可以处于数据框架中的任何 len (df)时间步骤。 如果在每个时间步骤中做出相同的选择，这意味着这个代理可以在相同的10个时间步骤中经历 len (df)30种可能的唯一状态。

While this may add quite a bit of noise to large data sets, I believe it should allow the agent to learn more from our limited amount of data. We will still traverse our test data set in serial fashion, to get a more accurate understanding of the algorithm’s usefulness on fresh, seemingly “live” data.

虽然这可能会给大型数据集增加相当多的噪音，但我相信它应该允许代理从我们有限的数据量中学到更多。 我们仍将以串行方式遍历测试数据集，以便更准确地理解算法对于新鲜的、看似"实时"的数据的有效性。

------

### Life Through The Agent’s Eyes

### 经纪人眼中的生活

It can often be helpful to visual an environment’s observation space, in order to get an idea of the types of features your agent will be working with. For example, here is a visualization of our observation space rendered using OpenCV.

它通常有助于可视化环境的观察空间，以便了解您的代理将使用的特性类型。 例如，下面是使用 OpenCV 渲染的我们的观察空间的可视化。



![img](https://cdn-images-1.medium.com/max/1600/1*DWO8W-DyghiZbgBstJZjWQ.gif)

OpenCV visualization of the environment’s observation space Opencv 环境观测空间的可视化

Each row in the image represents a row in our `observation_space`. The first 4 rows of frequency-like red lines represent the OHCL data, and the spurious orange and yellow dots directly below represent the volume. The fluctuating blue bar below that is the agent’s net worth, and the lighter blips below that represent the agent’s trades.

图像中的每一行代表我们观察空间中的一行。 前4行类似频率的红色线代表 OHCL 数据，而直接在下面的伪橙色和黄色点代表体积。 下面波动的蓝条代表经纪人的净资产，下面较小的光点代表经纪人的交易。

If you squint, you can just make out a candlestick graph, with volume bars below it and a strange morse-code like interface below that showing trade history. It looks like our agent should be able to learn sufficiently from the data in our `observation_space`, so let’s move on. Here we’ll define our `_next_observation` method, where we’ll scale the **observed data** from 0 to 1.

如果你眯起眼睛，你只能看到一个烛台图表，下面有数量条和一个奇怪的莫尔斯电码似的界面，显示了交易历史。 看起来我们的代理应该能够从我们的观测空间的数据中充分学习，所以让我们继续前进。 在这里，我们将定义下一个观测方法，我们将把观测数据从0放大到1。

> It’s important to only scale the data the agent has observed so far to prevent look-ahead biases. 为了避免前瞻性偏见，重要的是只扩展代理人迄今为止观察到的数据

```
def _next_observation(self):
  end = self.current_step + self.lookback_window_size + 1
  obs = np.array([
    self.active_df['Open'].values[self.current_step:end],  
    self.active_df['High'].values[self.current_step:end],
    self.active_df['Low'].values[self.current_step:end],
    self.active_df['Close'].values[self.current_step:end],
    self.active_df['Volume_(BTC)'].values[self.current_step:end],
  ])
  scaled_history = self.scaler.fit_transform(self.account_history)
  obs = np.append(obs, scaled_history[:, -(self.lookback_window_size
                                                     + 1):], axis=0)
  return obs
```

#### Taking Action

#### 采取行动

Now that we’ve set up our observation space, it’s time to write our `step`function, and in turn, take the agent’s prescribed action. Whenever `self.steps_left == 0` for our current trading session, we will sell any BTC we are holding and call `_reset_session()`. Otherwise, we set the `reward` to our current net worth and only set `done` to `True` if we’ve run out of money.

现在我们已经建立了我们的观察空间，是时候编写我们的步骤函数了，然后依次执行代理的规定操作。 每当 self.steps 为我们当前的交易时段离开0时，我们将卖出我们持有的任何 BTC，并称之为重置时段()。 否则，我们将奖励设置为我们当前的净资产，只有当我们用完钱时才设置为真。

```
def step(self, action):
  current_price = self._get_current_price() + 0.01
  self._take_action(action, current_price)
  self.steps_left -= 1
  self.current_step += 1
  if self.steps_left == 0:
    self.balance += self.btc_held * current_price
    self.btc_held = 0
    self._reset_session()
  obs = self._next_observation()
  reward = self.net_worth
  done = self.net_worth <= 0
  return obs, reward, done, {}
```

Taking an action is as simple as getting the `current_price`, determining the specified action, and either buying or selling the specified amount of BTC. Let’s quickly write `_take_action` so we can test our environment.

采取行动就像获取当前价格、确定指定的行动，以及买入或卖出指定数量的 BTC 一样简单。 让我们快速编写执行操作，以便测试我们的环境。

```
def _take_action(self, action, current_price):
  action_type = action[0]
  amount = action[1] / 10
  btc_bought = 0
  btc_sold = 0
  cost = 0
  sales = 0
  if action_type < 1:
    btc_bought = self.balance / current_price * amount
    cost = btc_bought * current_price * (1 + self.commission)
    self.btc_held += btc_bought
    self.balance -= cost
  elif action_type < 2:
    btc_sold = self.btc_held * amount
    sales = btc_sold * current_price  * (1 - self.commission)
    self.btc_held -= btc_sold
    self.balance += sales
```

Finally, in the same method, we will append the trade to `self.trades` and update our net worth and account history.

最后，使用同样的方法，我们将附加交易到 self.trades，并更新我们的净值和帐户历史。

```
  if btc_sold > 0 or btc_bought > 0:
    self.trades.append({
      'step': self.frame_start+self.current_step,
      'amount': btc_sold if btc_sold > 0 else btc_bought,
      'total': sales if btc_sold > 0 else cost,
      'type': "sell" if btc_sold > 0 else "buy"
    })
  self.net_worth = self.balance + self.btc_held * current_price
  self.account_history = np.append(self.account_history, [
    [self.net_worth],
    [btc_bought],
    [cost],
    [btc_sold],
    [sales]
  ], axis=1)
```

Our agents can now initiate a new environment, step through that environment, and take actions that affect the environment. It’s time to watch them trade.

我们的代理现在可以开始一个新的环境，逐步通过该环境，并采取影响环境的行动。 是时候看看他们的交易了。

------

### Watching Our Bots Trade

### 看着我们的机器人交易

Our `render` method could be something as simple as calling `print(self.net_worth)`, but that’s no fun. Instead we are going to plot a simple candlestick chart of the pricing data with volume bars and a separate plot for our net worth.

我们的 render 方法可以像调用 print (self. net worth)一样简单，但这并不有趣。 相反，我们将用数量 K线和我们的净资产单独绘制一张价格数据的简单图表。

We are going to take the code in `StockTradingGraph.py` from [the last article I wrote](https://medium.com/@notadamking/visualizing-stock-trading-agents-using-matplotlib-and-gym-584c992bc6d4), and re-purposing it to render our Bitcoin environment. You can grab the code from my [Github](https://github.com/notadamking/Stock-Trading-Visualization).

我们将从我上一篇文章中提取 stocktradinggraph.py 的代码，并重新利用它来呈现我们的比特币环境。 你可以从我的 Github 上获取代码。

The first change we are going to make is to update `self.df['Date']`everywhere to `self.df['Timestamp']`, and remove all calls to `date2num` as our dates already come in unix timestamp format. Next, in our `render` method we are going to update our date labels to print human-readable dates, instead of numbers.

我们要做的第一个更改是更新 self.df ['Date'] everywhere to self.df ['Timestamp'] ，并删除对 date2num 的所有调用，因为我们的日期已经采用 unix 时间戳格式。 接下来，在我们的渲染方法中，我们将更新日期标签来打印人类可读的日期，而不是数字。

```
from datetime import datetime
```

First, import the `datetime` library, then we’ll use the `utcfromtimestamp`method to get a UTC string from each timestamp and `strftime` to format the string in `Y-m-d H:M` format.

首先，导入日期时间库，然后使用 utcfromtimestamp 方法从每个时间戳和 strftime 获取 UTC 字符串，以 Y-m-d h: m 格式格式化该字符串。

```
date_labels = np.array([datetime.utcfromtimestamp(x).strftime(
'%Y-%m-%d %H:%M') for x in self.df['Timestamp'].values[step_range]])
```

Finally, we change `self.df['Volume']` to `self.df['Volume_(BTC)']` to match our data set, and we’re good to go. Back in our `BitcoinTradingEnv`, we can now write our `render` method to display the graph.

最后，我们将 self.df ['Volume']更改为 self.df ['Volume (BTC)']以匹配我们的数据集，然后就可以开始了。 回到 BitcoinTradingEnv，我们现在可以编写我们的 render 方法来显示图形。

```
def render(self, mode='human', **kwargs):
  if mode == 'human':
    if self.viewer == None:
      self.viewer = BitcoinTradingGraph(self.df,
                                        kwargs.get('title', None))
    self.viewer.render(self.frame_start + self.current_step,
                       self.net_worth,
                       self.trades,
                       window_size=self.lookback_window_size)
```

And voila! We can now watch our agents trade Bitcoin.

瞧！ 我们现在可以看到我们的代理商交易比特币。



![img](https://cdn-images-1.medium.com/max/1600/1*f8gvwrKvpij6m-KCL6wxGA.gif)

Matplotlib visualization of our agent trading Bitcoin 我们的代理交易比特币的 Matplotlib 可视化

The green ghosted tags represent buys of BTC and the red ghosted tags represent sells. The white tag on the top right is the agent’s current net worth and the bottom right tag is the current price of Bitcoin. Simple, yet elegant. Now, it’s time to train our agent and see how much money we can make!

绿色幽灵标签代表购买 BTC，红色幽灵标签代表销售。 右上角的白色标签是代理商当前的净资产，右下角的标签是比特币当前的价格。 简单，但优雅。 现在，是时候训练我们的代理商了，看看我们能赚多少钱！

#### Training Time

#### 训练时间

One of the criticisms I received on my first article was the lack of cross-validation, or splitting the data into a training set and test set. The purpose of doing this is to test the accuracy of your final model on fresh data it has never seen before. While this was not a concern of that article, it definitely is here. Since we are using time series data, we don’t have many options when it comes to cross-validation.

在我的第一篇文章中，我收到的批评之一是缺乏交叉验证，或者将数据分割成训练集和测试集。 这样做的目的是在以前从未见过的新数据上测试最终模型的准确性。 虽然这不是那篇文章关心的问题，但它肯定在这里。 因为我们使用的是时间序列数据，所以当涉及到交叉验证时我们没有太多的选择。

For example, one common form of cross validation is called *k-fold* validation, in which you split the data into k equal groups and one by one single out a group as the test group and use the rest of the data as the training group. However time series data is highly time dependent, meaning later data is highly dependent on previous data. So k-fold won’t work, because our agent will learn from future data before having to trade it, an unfair advantage.

例如，一种常见的交叉验证形式被称为 k-fold validation，在这种验证中，你将数据分成 k 个相等的组，一个一个的分出一个组作为测试组，并使用其余的数据作为训练组。 然而，时间序列数据高度依赖于时间，这意味着后来的数据高度依赖于先前的数据。 所以 k-fold 不会起作用，因为我们的代理人在交易之前会从未来的数据中学习，这是一种不公平的优势。

This same flaw applies to most other cross-validation strategies when applied to time series data. So we are left with simply taking a slice of the full data frame to use as the training set from the beginning of the frame up to some arbitrary index, and using the rest of the data as the test set.

这一缺陷同样适用于大多数其他交叉验证 / 时间序列数据策略。 所以我们只需要简单地从整个数据框架的一个片段作为从框架开始到任意索引的训练集，并使用其余的数据作为测试集。

```
slice_point = int(len(df) - 100000)
train_df = df[:slice_point]
test_df = df[slice_point:]
```

Next, since our environment is only set up to handle a single data frame, we will create two environments, one for the training data and one for the test data.

接下来，由于我们的环境只设置为处理单个数据帧，因此我们将创建两个环境，一个用于培训数据，另一个用于测试数据。

```
train_env = DummyVecEnv([lambda: BitcoinTradingEnv(train_df, 
                         commission=0, serial=False)])
test_env = DummyVecEnv([lambda: BitcoinTradingEnv(test_df, 
                        commission=0, serial=True)])
```

Now, training our model is as simple as creating an agent with our environment and calling `model.learn`.

现在，训练我们的模型就像在环境中创建一个代理并调用 model.learn 一样简单。

```
model = PPO2(MlpPolicy,
             train_env,
             verbose=1, 
             tensorboard_log="./tensorboard/")
model.learn(total_timesteps=50000)
```

Here, we are using tensorboard so we can easily visualize our tensorflow graph and view some quantitative metrics about our agents. For example, here is a graph of the discounted rewards of many agents over 200,000 time steps:

在这里，我们使用的是张量板，因此我们可以很容易地可视化我们的张量流图，并查看一些关于我们的代理的定量度量。 例如，这里有一个图表，显示了许多代理商超过200,000个时间步骤的折扣报酬:



![img](https://cdn-images-1.medium.com/max/1600/1*C3Z4y4EUeN8mLpmdbLPUZA.png)

Wow, it looks like our agents are extremely profitable! Our best agent was even capable of 1000x’ing his balance over the course of 200,000 steps, and the rest averaged at least a 30x increase!

哇，看来我们的代理商利润非常可观！ 我们最好的经纪人甚至可以在200,000步的过程中增加1000倍的平衡，其余的平均增加至少30倍！

It was at this point that I realized there was a bug in the environment… Here is the new rewards graph, after fixing that bug:

正是在这个时候，我意识到环境中有一个 bug... ... 在修复了这个 bug 之后，这里有一个新的奖励图表:



![img](https://cdn-images-1.medium.com/max/1600/1*SFNha2nSRaeE100dTCIXLQ.png)

As you can see, a couple of our agents did well, and the rest traded themselves into bankruptcy. However, the agents that did well were able to 10x and even 60x their initial balance, at best. I must admit, all of the profitable agents were trained and tested in an environment without commissions, so it is still entirely unrealistic for our agent’s to make any *real money.* But we’re getting somewhere!

正如你所看到的，我们的一些代理商做得很好，而其他的代理商则以破产交易。 然而，那些表现出色的代理商最多只能达到初始余额的10倍甚至60倍。 我必须承认，所有有利可图的代理商都是在一个没有佣金的环境中接受培训和测试的，所以我们的代理商想要赚到真金白银是完全不现实的。 但是我们有进展了！

Let’s test our agents on the test environment (with fresh data they’ve never seen before), to see how well they’ve learned to trade Bitcoin.

让我们在测试环境中测试我们的代理(使用他们从未见过的新数据) ，看看他们在交易比特币方面学得有多好。



![img](https://cdn-images-1.medium.com/max/1600/1*UCtL7UMAHKnx4ePoP-0p2w.png)

Our trained agents race to bankruptcy when trading on fresh, test data 我们训练有素的特工在交易新的测试数据时竞相破产

Clearly, we’ve still got quite a bit of work to do. By simply switching our model to use stable-baseline’s A2C, instead of the current PPO2 agent, we can greatly improve our performance on this data set. Finally, we can update our reward function slightly, as per [Sean O’Gorman’s advice](https://medium.com/@SOGorman35/now-that-i-had-a-chance-to-read-your-article-in-a-bit-more-depth-ill-add-some-more-input-beyond-b71e442bb8a), so that we reward increases in net worth, not just achieving a high net worth and staying there.

显然，我们还有很多工作要做。 通过简单地切换我们的模型使用稳定基线的 A2C，而不是当前的 PPO2代理，我们可以大大提高我们在这个数据集上的性能。 最后，我们可以根据肖恩 · 奥 · 戈尔曼的建议，稍微更新一下我们的奖励功能，这样我们就可以奖励净资产的增加，而不仅仅是获得高净资产并保持在那里。

```
reward = self.net_worth - prev_net_worth
```

These two changes alone greatly improve the performance on the same data set, and as you can see below, we are finally able to achieve profitability on fresh data that wasn’t in the training set.

这两个变化本身就极大地提高了相同数据集上的性能，正如您在下面看到的，我们终于能够在培训集之外的新数据上实现盈利。



![img](https://cdn-images-1.medium.com/max/1600/1*wz5XAg-8PYRDmzBMKdakHw.png)

However, we can do much better. In order for us to improve these results, we are going to need to optimize our hyper-parameters and train our agents for much longer. Time to break out the GPU and get to work!

然而，我们可以做得更好。 为了改善这些结果，我们需要优化我们的超参数和训练我们的代理人更长的时间。 是时候打破 GPU，开始工作了！

However, this article is already a bit long and we’ve still got quite a bit of detail to go over, so we are going to take a break here. In the next article, we will use [**Bayesian optimization**](https://arxiv.org/abs/1807.02811) to zone in on the best hyper-parameters for our problem space, and prepare the environment for training/testing on GPUs using CUDA.

然而，这篇文章已经有点长了，我们还有相当多的细节要过一遍，所以我们将在这里休息一下。 在下一篇文章中，我们将使用贝叶斯优化来区分我们的问题空间的最佳超参数，并准备使用 CUDA 在 gpu 上进行培训 / 测试的环境。

------

### Conclusion

### 总结

In this article, we set out to create a profitable Bitcoin trading agent from scratch, using deep reinforcement learning. We were able to accomplish the following:

在这篇文章中，我们开始创建一个利润丰厚的比特币交易代理从零开始，使用深强化学习。 我们能够完成以下工作:

1. Created a Bitcoin trading environment from scratch using OpenAI’s gym. 使用 OpenAI 的健身房从零开始创建一个比特币交易环境
2. Built a visualization of that environment using Matplotlib. 使用 Matplotlib 构建该环境的可视化
3. Trained and tested our agents using simple cross-validation. 训练和测试我们的特工使用简单的交叉验证
4. Realized we still need a lot of work, but we can see the light at the end of the tunnel 意识到我们仍然需要大量的工作，但我们可以看到隧道尽头的光

While we still haven’t quite succeeded in making a profitable Bitcoin trading bot *on fresh data,* we are much closer to our goal than when we set out. Next time, we will make sure our agent can make some money on the test data, not just the training data. Stay tuned for my next article, and long live Bitcoin!

虽然我们还没有成功地在新数据上制造出一个盈利的比特币交易机器人，但我们已经比刚开始的时候更接近目标了。 下一次，我们将确保我们的代理人可以在测试数据上赚一些钱，而不仅仅是培训数据。 请继续关注我的下一篇文章，比特币万岁！

------

*Thanks for reading! As always, all of the code for this tutorial can be found on my* [*Github*](https://github.com/notadamking/Bitcoin-Trader-RL)*. Leave a comment below if you have any questions or feedback, I’d love to hear from you! I can also be reached on* [*Twitter*](https://twitter.com/notadamking) *at @notadamking.*

谢谢阅读！ 和往常一样，本教程的所有代码都可以在我的 Github 上找到。 如果你有任何问题或反馈，请在下面留言，我很乐意听取你的意见！ 我也可以在 Twitter 上联系到@notadamking。

