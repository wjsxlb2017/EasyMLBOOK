# 图文并茂的Transformer

在之前的文章中，我们研究了注意力——一种在现代深度学习模型中普遍存在的方法。 注意力是一个有助于提高神经机器翻译应用性能的概念。 在这篇文章中，我们将看看Transformer模型，利用注意力提高速度，这些模型可以训练。 在特定的任务中，Transformer翻译的表现优于谷歌神经机器翻译模型。 然而，最大的好处来自于 Transformer可以并行化。 事实上，Google Cloud 建议使用 The Transformer 作为参考模型来使用他们的 Cloud TPU 产品。 那么，让我们试着把这个模型拆开，看看它是如何工作的。


Transformer是在论文中提出的<关注是所有你需要的>。 它的 TensorFlow 实现可以作为  [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)包的一部分。 在这篇文章中，我们将尝试把事情简单化，并逐一介绍这些概念，希望这样可以让那些对主题没有深入了解的人们更容易理解。

## A High-Level Look


让我们首先把模型看作一个黑盒子。 在机器翻译应用程序中，它会将一个句子翻译成一种语言，然后再将其翻译成另一种语言。

![img](https://jalammar.github.io/images/t/the_transformer_3.png)


打开擎天柱，我们看到一个encoding组件，一个decoding组件，以及它们之间的连接。

![img](https://jalammar.github.io/images/t/The_transformer_encoders_decoders.png)


encoding组件是一堆encoding(论文堆了六个)。 encoding组件是一组数目相同的解码器。

![img](https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)


encoders在结构上都是相同的(但是它们没有共享权重)。 每个子层分为两个子层:

![img](https://jalammar.github.io/images/t/Transformer_encoder.png)

The encoder’s inputs first flow through a self-attention layer – a layer that helps the encoder look at other words in the input sentence as it encodes a specific word. We’ll look closer at self-attention later in the post.

编码器的输入首先通过一个自我注意层-一个层，帮助编码器看看在输入句子中的其他单词，因为它编码一个特定的单词。 我们将在稍后的文章中进一步研究自我关注。

The outputs of the self-attention layer are fed to a feed-forward neural network. The exact same feed-forward network is independently applied to each position.

自注意层的输出被反馈给前馈神经网络。 完全相同的前馈网络独立地应用于每个位置。

The decoder has both those layers, but between them is an attention layer that helps the decoder focus on relevant parts of the input sentence (similar what attention does in [seq2seq models](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)).

解码器具有这两个层次，但是它们之间有一个注意力层，帮助解码器集中注意输入句子的相关部分(类似于 seq2seq 模型中的注意力)。

![img](https://jalammar.github.io/images/t/Transformer_decoder.png)

## Bringing The Tensors Into The Picture

## 将张量引入画面

Now that we’ve seen the major components of the model, let’s start to look at the various vectors/tensors and how they flow between these components to turn the input of a trained model into an output.

现在我们已经了解了模型的主要组成部分，让我们开始看看各种矢量 / 张量，以及它们如何在这些组件之间流动，从而将经过训练的模型的输入转化为输出。

As is the case in NLP applications in general, we begin by turning each input word into a vector using an [embedding algorithm](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca).

和一般的 NLP 应用程序一样，我们首先使用嵌入算法将每个输入单词转换成一个向量。



![img](https://jalammar.github.io/images/t/embeddings.png) 
Each word is embedded into a vector of size 512. We'll represent those vectors with these simple boxes. 每个单词都嵌入一个大小为512的矢量中。 我们用这些简单的盒子来表示这些矢量

The embedding only happens in the bottom-most encoder. The abstraction that is common to all the encoders is that they receive a list of vectors each of the size 512 – In the bottom encoder that would be the word embeddings, but in other encoders, it would be the output of the encoder that’s directly below. The size of this list is hyperparameter we can set – basically it would be the length of the longest sentence in our training dataset.

嵌入只发生在最底层的编码器中。 所有编码器共同的抽象是，他们接收到一个大小为512的向量列表——在底部编码器中是字嵌入，但在其他编码器中，它是直接在下面的编码器的输出。 这个列表的大小是我们可以设置的超参数——基本上就是我们训练数据集中最长的句子的长度。

After embedding the words in our input sequence, each of them flows through each of the two layers of the encoder.

在我们的输入序列中嵌入单词之后，每个单词流经编码器的两个层中的每一层。

![img](https://jalammar.github.io/images/t/encoder_with_tensors.png)

Here we begin to see one key property of the Transformer, which is that the word in each position flows through its own path in the encoder. There are dependencies between these paths in the self-attention layer. The feed-forward layer does not have those dependencies, however, and thus the various paths can be executed in parallel while flowing through the feed-forward layer.

在这里，我们开始看到 Transformer 的一个关键属性，即每个位置中的单词通过编码器中它自己的路径流动。 在自我注意层，这些路径之间存在依赖关系。 但是，前向层没有这些依赖项，因此可以在通过前向层时并行执行各种路径。

Next, we’ll switch up the example to a shorter sentence and we’ll look at what happens in each sub-layer of the encoder.

接下来，我们将把这个例子切换到一个较短的句子，我们将看看编码器的每个子层中发生了什么。

## Now We’re Encoding!

## 现在我们正在编码

As we’ve mentioned already, an encoder receives a list of vectors as input. It processes this list by passing these vectors into a ‘self-attention’ layer, then into a feed-forward neural network, then sends out the output upwards to the next encoder.

正如我们已经提到的，编码器接收作为输入的向量列表。 它处理这个列表通过传递这些矢量到一个"自我注意"层，然后进入一个前馈神经网络，然后发送输出向上到下一个编码器。

![img](https://jalammar.github.io/images/t/encoder_with_tensors_2.png)
The word at each position passes through a self-encoding process. Then, they each pass through a feed-forward neural network -- the exact same network with each vector flowing through it separately. 位于每个位置的单词通过一个自编码过程。 然后，它们各自通过一个前馈神经网络——一个完全相同的网络，每个矢量分别通过它

## Self-Attention at a High Level

## 高层次的自我注意力

Don’t be fooled by me throwing around the word “self-attention” like it’s a concept everyone should be familiar with. I had personally never came across the concept until reading the Attention is All You Need paper. Let us distill how it works.

不要被我用"自我关注"这个词糊弄了，好像这是一个每个人都应该熟悉的概念。 我个人从来没有遇到过这个概念，直到阅读的关注是所有你需要的文件。 让我们提炼一下它是如何工作的。

Say the following sentence is an input sentence we want to translate:

假设下面的句子是我们要翻译的输入句子:

”`The animal didn't cross the street because it was too tired`”

"那只动物因为太累了而没有过马路"

What does “it” in this sentence refer to? Is it referring to the street or to the animal? It’s a simple question to a human, but not as simple to an algorithm.

这个句子中的"it"指的是什么？ 它指的是街道还是动物？ 对于人类来说，这是一个简单的问题，但对于算法来说就不那么简单了。

When the model is processing the word “it”, self-attention allows it to associate “it” with “animal”.

当模型处理单词"it"时，自我注意使它能够将"it"与"animal"联系起来。

As the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word.

当模型处理每个单词(输入序列中的每个位置)时，自我注意允许它查看输入序列中的其他位置，以寻找线索，从而帮助更好地编码这个单词。

If you’re familiar with RNNs, think of how maintaining a hidden state allows an RNN to incorporate its representation of previous words/vectors it has processed with the current one it’s processing. Self-attention is the method the Transformer uses to bake the “understanding” of other relevant words into the one we’re currently processing.

如果你熟悉 RNN，想想如何维护一个隐藏状态允许一个 RNN 合并它处理的当前词 / 向量的先前词 / 向量的表示。 自我关注是变压器用来把其他相关单词的"理解"融入到我们正在处理的单词中的一种方法。

![img](https://jalammar.github.io/images/t/transformer_self-attention_visualization.png) 
As we are encoding the word "it" in encoder #5 (the top encoder in the stack), part of the attention mechanism was focusing on "The Animal", and baked a part of its representation into the encoding of "it". 当我们将"it"这个词编码在编码器 # 5(堆栈顶部的编码器)中时，注意力机制的一部分集中在"The Animal"上，并将它的一部分表示融入了"it"的编码中

Be sure to check out the [Tensor2Tensor notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) where you can load a Transformer model, and examine it using this interactive visualization.

请务必查看 Tensor2Tensor 笔记本，在其中您可以加载 Transformer 模型，并使用这个交互式可视化检查它。

## Self-Attention in Detail

## 细节上的自我关注

Let’s first look at how to calculate self-attention using vectors, then proceed to look at how it’s actually implemented – using matrices.

让我们先看看如何使用向量计算自我注意力，然后再看看它实际上是如何实现的——使用矩阵。

The **first step** in calculating self-attention is to create three vectors from each of the encoder’s input vectors (in this case, the embedding of each word). So for each word, we create a Query vector, a Key vector, and a Value vector. These vectors are created by multiplying the embedding by three matrices that we trained during the training process.

计算自我注意力的第一步是从编码器的每个输入向量中创建三个向量(在本例中是每个单词的嵌入)。 因此，对于每个单词，我们创建一个 Query vector、一个 Key vector 和一个 Value vector。 这些向量是通过将嵌入乘以我们在训练过程中训练的三个矩阵来创建的。

Notice that these new vectors are smaller in dimension than the embedding vector. Their dimensionality is 64, while the embedding and encoder input/output vectors have dimensionality of 512. They don’t HAVE to be smaller, this is an architecture choice to make the computation of multiheaded attention (mostly) constant.

请注意，这些新矢量的维数比嵌入矢量小。 其维数为64，而嵌入和编码输入输出向量的维数为512。 它们不需要更小，这是一个使多头注意力(主要是)的计算恒定的架构选择。



![img](https://jalammar.github.io/images/t/transformer_self_attention_vectors.png) 
Multiplying 成倍增长x1 by the 通过WQ weight matrix produces 权矩阵产生q1 第一季度, the "query" vector associated with that word. We end up creating a "query", a "key", and a "value" projection of each word in the input sentence. ，与该单词相关的"查询"向量。 我们最终在输入句子中为每个单词创建一个"query"、一个"key"和一个"value"投影





What are the “query”, “key”, and “value” vectors? 

They’re abstractions that are useful for calculating and thinking about attention. Once you proceed with reading how attention is calculated below, you’ll know pretty much all you need to know about the role each of these vectors plays.

什么是"查询"、"键"和"值"向量？ 它们是抽象的，对于计算和思考注意力是有用的。 一旦你继续阅读下面关于注意力是如何计算的，你就会知道关于这些向量所扮演的角色的所有你需要知道的东西。

The **second step** in calculating self-attention is to calculate a score. Say we’re calculating the self-attention for the first word in this example, “Thinking”. We need to score each word of the input sentence against this word. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position.

计算自我注意力的第二步是计算分数。 假设我们正在计算这个例子中第一个单词"Thinking"的自我注意力。 我们需要在输入句子的每个单词上对这个单词打分。 分数决定了当我们在某个位置编码一个单词时，对输入句子其他部分的关注程度。

The score is calculated by taking the dot product of the query vector with the key vector of the respective word we’re scoring. So if we’re processing the self-attention for the word in position #1, the first score would be the dot product of q1 and k1. The second score would be the dot product of q1 and k2.

计算得分的方法是将查询向量的点积与我们要计分的相应单词的关键向量相乘。 所以如果我们处理位置 # 1中单词的自我注意，第一个得分是 q1和 k1的点积。 第二个得分是 q1和 k2的点积。



![img](https://jalammar.github.io/images/t/transformer_self_attention_score.png) 



The **third and forth steps** are to divide the scores by 8 (the square root of the dimension of the key vectors used in the paper – 64. This leads to having more stable gradients. There could be other possible values here, but this is the default), then pass the result through a softmax operation. Softmax normalizes the scores so they’re all positive and add up to 1.

第三步和第四步是将分数除以8(论文中使用的关键向量维数的平方根)。 这导致了更稳定的梯度。 这里可能还有其他可能的值，但这是默认值) ，然后通过 softmax 操作传递结果。 软最大的正常化的分数，所以他们都是积极的，加起来为1。



![img](https://jalammar.github.io/images/t/self-attention_softmax.png) 

This softmax score determines how much how much each word will be expressed at this position. Clearly the word at this position will have the highest softmax score, but sometimes it’s useful to attend to another word that is relevant to the current word.

这个最大分数决定了每个单词在这个位置上的表达量。 很明显，这个位置上的单词会有最高的最低分，但是有时候注意与当前单词相关的另一个单词是很有用的。



The **fifth step** is to multiply each value vector by the softmax score (in preparation to sum them up). The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny numbers like 0.001, for example).

第五步是将每个值向量乘以软最大得分(准备将它们加起来)。 这里的直觉是保持我们想要关注的单词的价值不变，屏蔽掉不相关的单词(例如，用0.001这样的小数字乘以它们)。

The **sixth step** is to sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word).

第六步是对加权值向量进行求和。 这将在这个位置产生自我注意层的输出(对于第一个单词)。



![img](https://jalammar.github.io/images/t/self-attention-output.png) 

That concludes the self-attention calculation. The resulting vector is one we can send along to the feed-forward neural network. In the actual implementation, however, this calculation is done in matrix form for faster processing. So let’s look at that now that we’ve seen the intuition of the calculation on the word level.

这就结束了自我关注的计算。 由此产生的矢量是一个我们可以发送到前馈神经网络。 然而，在实际实现中，这种计算是以矩阵形式进行的，以便更快地进行处理。 现在让我们来看看，我们已经看到了单词水平的计算的直觉。

## Matrix Calculation of Self-Attention

## 自我注意的矩阵计算

**The first step** is to calculate the Query, Key, and Value matrices. We do that by packing our embeddings into a matrix X, and multiplying it by the weight matrices we’ve trained (WQ, WK, WV).

第一步是计算 Query、 Key 和 Value 矩阵。 我们通过将我们的嵌入包装到一个矩阵 x 中，并将其乘以我们训练的权重矩阵(WQ、 WK、 WV)来实现这一点。

![img](https://jalammar.github.io/images/t/self-attention-matrix-calculation.png) 
Every row in the 中的每一行X matrix corresponds to a word in the input sentence. We again see the difference in size of the embedding vector (512, or 4 boxes in the figure), and the q/k/v vectors (64, or 3 boxes in the figure) 矩阵对应于输入句子中的一个单词。 我们再次看到嵌入向量(图中为512或4个方框)和 q / k / v 向量(图中为64或3个方框)的大小差异



**Finally**, since we’re dealing with matrices, we can condense steps two through six in one formula to calculate the outputs of the self-attention layer.

最后，因为我们处理的是矩阵，我们可以浓缩步骤二到六在一个公式计算出输出的自我注意层。

![img](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png) 
The self-attention calculation in matrix form 矩阵形式的自注意计算





## The Beast With Many Heads

## 多头怪兽

The paper further refined the self-attention layer by adding a mechanism called “multi-headed” attention. This improves the performance of the attention layer in two ways:

论文通过增加"多头"注意机制进一步细化了自我注意层。 这在两个方面改善了注意力层的表现:

1. It expands the model’s ability to focus on different positions. Yes, in the example above, z1 contains a little bit of every other encoding, but it could be dominated by the the actual word itself. It would be useful if we’re translating a sentence like “The animal didn’t cross the street because it was too tired”, we would want to know which word “it” refers to.

   它扩展了模型关注不同位置的能力。 是的，在上面的例子中，z1包含了其他所有编码的一小部分，但是它可以被实际的单词本身控制。 如果我们翻译一个句子，比如"The animal didn't cross The street because It was too tired"，我们会想知道"It"指的是哪个词。

2. It gives the attention layer multiple “representation subspaces”. As we’ll see next, with multi-headed attention we have not only one, but multiple sets of Query/Key/Value weight matrices (the Transformer uses eight attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.

   它给予注意层多个"表示子空间"。 正如我们接下来将看到的，通过多重注意，我们不仅有一组，而且有多组 query / key / value 权重矩阵(Transformer 使用八个注意力头，因此我们最终为每个编码器 / 解码器设置了八组)。 这些集合中的每一个都是随机初始化的。 然后，在训练之后，每个集合用于将输入嵌入(或来自低级编码器 / 译码器的向量)投影到一个不同的表示子空间。

![img](https://jalammar.github.io/images/t/transformer_attention_heads_qkv.png)
With multi-headed attention, we maintain separate Q/K/V weight matrices for each head resulting in different Q/K/V matrices. As we did before, we multiply X by the WQ/WK/WV matrices to produce Q/K/V matrices. 在多头注意的情况下，我们对每个头维持不同的 q / k / v 权矩阵，得到不同的 q / k / v 矩阵。 如前所述，我们将 x 乘以 wq / wk / wv 矩阵，得到 q / k / v 矩阵


If we do the same self-attention calculation we outlined above, just eight different times with different weight matrices, we end up with eight different Z matrices

如果我们做上面提到的相同的自我注意计算，只是用不同的权重矩阵进行8次不同的计算，我们就得到了8个不同的 z 矩阵

![img](https://jalammar.github.io/images/t/transformer_attention_heads_z.png)



This leaves us with a bit of a challenge. The feed-forward layer is not expecting eight matrices – it’s expecting a single matrix (a vector for each word). So we need a way to condense these eight down into a single matrix.

这给我们带来了一些挑战。 前馈层不需要八个矩阵——它需要一个单一的矩阵(每个单词都有一个矢量)。 所以我们需要一种方法，把这八个元素浓缩成一个单一的矩阵。

How do we do that? We concat the matrices then multiple them by an additional weights matrix WO.

我们该怎么做呢？ 然后用一个加权矩阵 WO 将它们相乘。

![img](https://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png)

That’s pretty much all there is to multi-headed self-attention. It’s quite a handful of matrices, I realize. Let me try to put them all in one visual so we can look at them in one place

这几乎就是多头自我关注的全部内容。 我意识到这是一大堆矩阵。 让我试着把它们放在一个视觉上，这样我们就可以在一个地方看到它们



![img](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)



Now that we have touched upon attention heads, let’s revisit our example from before to see where the different attention heads are focusing as we encode the word “it” in our example sentence:

既然我们已经提到了注意力，让我们回顾一下之前的例子，看看当我们在例句中编码单词"it"时，不同的注意力集中在哪里:

![img](https://jalammar.github.io/images/t/transformer_self-attention_visualization_2.png) 
As we encode the word "it", one attention head is focusing most on "the animal", while another is focusing on "tired" -- in a sense, the model's representation of the word "it" bakes in some of the representation of both "animal" and "tired". 在我们对"it"这个词进行编码时，一个注意力集中在"动物"上，而另一个注意力集中在"疲劳"上——从某种意义上说，这个模型对"it"这个词的表达既是"动物"的表达，也是"疲劳"的表达



If we add all the attention heads to the picture, however, things can be harder to interpret:

然而，如果我们把所有的注意力都集中在图片上，事情就会变得更难理解:

![img](https://jalammar.github.io/images/t/transformer_self-attention_visualization_3.png) 

## Representing The Order of The Sequence Using Positional Encoding

## 用位置编码表示序列的顺序

One thing that’s missing from the model as we have described it so far is a way to account for the order of the words in the input sequence.

到目前为止，我们所描述的模型中缺少的一个东西是一种考虑输入序列中单词顺序的方法。

To address this, the transformer adds a vector to each input embedding. These vectors follow a specific pattern that the model learns, which helps it determine the position of each word, or the distance between different words in the sequence. The intuition here is that adding these values to the embeddings provides meaningful distances between the embedding vectors once they’re projected into Q/K/V vectors and during dot-product attention.

为了解决这个问题，转换器向每个输入嵌入添加一个向量。 这些向量遵循模型学习的特定模式，这有助于模型确定每个单词的位置，或序列中不同单词之间的距离。 这里的直觉是，将这些值添加到嵌入中，一旦嵌入向量被投射到 q / k / v 向量中，以及在点积注意期间，就可以在嵌入向量之间提供有意义的距离。



![img](https://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png)
To give the model a sense of the order of the words, we add positional encoding vectors -- the values of which follow a specific pattern. 为了让模型感觉到单词的顺序，我们添加了位置编码向量——它们的值遵循一个特定的模式



If we assumed the embedding has a dimensionality of 4, the actual positional encodings would look like this:

如果我们假设嵌入的维数是4，那么实际的位置编码看起来是这样的:

![img](https://jalammar.github.io/images/t/transformer_positional_encoding_example.png)
A real example of positional encoding with a toy embedding size of 4 4的玩具嵌入大小的位置编码的一个实例



What might this pattern look like?

这个模式看起来像什么？

In the following figure, each row corresponds the a positional encoding of a vector. So the first row would be the vector we’d add to the embedding of the first word in an input sequence. Each row contains 512 values – each with a value between 1 and -1. We’ve color-coded them so the pattern is visible.

在下图中，每一行对应一个向量的位置编码。 所以第一行是向量，我们要把第一个单词嵌入到输入序列中。 每行包含512个值，每个值介于1和 -1之间。 我们对它们进行了颜色编码，所以图案是可见的。

![img](https://jalammar.github.io/images/t/transformer_positional_encoding_large_example.png)
A real example of positional encoding for 20 words (rows) with an embedding size of 512 (columns). You can see that it appears split in half down the center. That's because the values of the left half are generated by one function (which uses sine), and the right half is generated by another function (which uses cosine). They're then concatenated to form each of the positional encoding vectors. 20个字(行)的位置编码的一个实际例子，嵌入大小为512(列)。 你可以看到它在中间分成了两半。 这是因为左半边的值是由一个函数(使用正弦函数)生成的，而右半边的值是由另一个函数(使用余弦函数)生成的。 然后将它们串联起来，形成每个位置编码向量

The formula for positional encoding is described in the paper (section 3.5). You can see the code for generating positional encodings in [`get_timing_signal_1d()`](https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py). This is not the only possible method for positional encoding. It, however, gives the advantage of being able to scale to unseen lengths of sequences (e.g. if our trained model is asked to translate a sentence longer than any of those in our training set).

本文(3.5节)描述了位置编码的公式。 您可以在 get 计时信号1d ()中看到用于生成位置编码的代码。 这不是位置编码的唯一可能的方法。 然而，它的优点是能够扩展到看不见的长度的序列(例如，如果我们的训练模型被要求翻译一个比我们训练集中的任何一个句子都长的句子)。

## The Residuals

## 残差

One detail in the architecture of the encoder that we need to mention before moving on, is that each sub-layer (self-attention, ffnn) in each encoder has a residual connection around it, and is followed by a [layer-normalization](https://arxiv.org/abs/1607.06450) step.

在我们继续之前需要提到的编码器体系结构中的一个细节是，每个编码器中的每个子层(自我注意，ffnn)都有一个围绕它的残余连接，然后是一个层标准化步骤。

![img](https://jalammar.github.io/images/t/transformer_resideual_layer_norm.png) 

If we’re to visualize the vectors and the layer-norm operation associated with self attention, it would look like this:

如果我们将向量和与自我注意相关的层规范操作可视化，它看起来像这样:

![img](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_2.png) 

This goes for the sub-layers of the decoder as well. If we’re to think of a Transformer of 2 stacked encoders and decoders, it would look something like this:

这也适用于解码器的子层。 如果我们想到一个由两个堆叠编码器和解码器组成的变压器，它看起来应该是这样的:

![img](https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png)

## The Decoder Side

## 解码者的一面

Now that we’ve covered most of the concepts on the encoder side, we basically know how the components of decoders work as well. But let’s take a look at how they work together.

现在我们已经涵盖了编码器端的大部分概念，我们基本上知道了解码器的组件是如何工作的。 但是让我们来看看它们是如何协同工作的。

The encoder start by processing the input sequence. The output of the top encoder is then transformed into a set of attention vectors K and V. These are to be used by each decoder in its “encoder-decoder attention” layer which helps the decoder focus on appropriate places in the input sequence:

编码器通过处理输入序列启动。 顶部编码器的输出然后转换成一组注意力向量 k 和 v。 每个解码器在其「编码器注意力」层使用这些功能，以协助解码器注意输入序列中的适当位置:

![img](https://jalammar.github.io/images/t/transformer_decoding_1.gif)
After finishing the encoding phase, we begin the decoding phase. Each step in the decoding phase outputs an element from the output sequence (the English translation sentence in this case). 完成编码阶段后，我们开始解码阶段。 解码阶段的每个步骤都从输出序列(本例中为英语翻译句)中输出一个元素

The following steps repeat the process until a special symbol is reached indicating the transformer decoder has completed its output. The output of each step is fed to the bottom decoder in the next time step, and the decoders bubble up their decoding results just like the encoders did. And just like we did with the encoder inputs, we embed and add positional encoding to those decoder inputs to indicate the position of each word.

以下步骤重复此过程，直到到达一个特殊符号，表示变压器解码器已完成其输出。 每个步骤的输出在下一个时间步骤中被输入到底层解码器，解码器会像编码器一样将解码结果显示出来。 就像我们对编码器输入所做的一样，我们在这些解码器输入中嵌入并加入位置编码来指示每个单词的位置。

![img](https://jalammar.github.io/images/t/transformer_decoding_2.gif)

The self attention layers in the decoder operate in a slightly different way than the one in the encoder:

解码器中的自我注意层与编码器中的自我注意层运作方式略有不同:

In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence. This is done by masking future positions (setting them to `-inf`) before the softmax step in the self-attention calculation.

在解码器中，自注意层仅允许注意输出序列中的早期位置。 这是通过屏蔽未来的位置(设置他们为 -inf)之前的软件最大步骤在自我注意力计算。

The “Encoder-Decoder Attention” layer works just like multiheaded self-attention, except it creates its Queries matrix from the layer below it, and takes the Keys and Values matrix from the output of the encoder stack.

"编码器-解码器注意"层的工作原理和多头自我注意一样，只不过它从下面的层创建查询矩阵，并从编码器栈的输出中获取键和值矩阵。

## The Final Linear and Softmax Layer

## 最后的线性和软最大层

The decoder stack outputs a vector of floats. How do we turn that into a word? That’s the job of the final Linear layer which is followed by a Softmax Layer.

解码器栈输出一个浮点向量。 我们怎么把它变成一个词呢？ 这是最后的线性层的工作，其次是软最大层。

The Linear layer is a simple fully connected neural network that projects the vector produced by the stack of decoders, into a much, much larger vector called a logits vector.

线性层是一个简单的完全连接的神经网络，它将解码器堆叠产生的矢量投射到一个更大的称为对数矢量的矢量中。

Let’s assume that our model knows 10,000 unique English words (our model’s “output vocabulary”) that it’s learned from its training dataset. This would make the logits vector 10,000 cells wide – each cell corresponding to the score of a unique word. That is how we interpret the output of the model followed by the Linear layer.

假设我们的模型知道10,000个独特的英语单词(我们的模型的"输出词汇表") ，这些单词是从它的训练数据集中学习的。 这将使 logits 矢量10,000个单元格宽——每个单元格对应一个唯一单词的得分。 这就是我们如何解释模型的输出，其次是线性层。

The softmax layer then turns those scores into probabilities (all positive, all add up to 1.0). The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.

最大软件层然后将这些分数转换成概率(全部为正，所有加起来为1.0)。 选择概率最高的单元格，并生成与其相关的单词作为此时间步骤的输出。



![img](https://jalammar.github.io/images/t/transformer_decoder_output_softmax.png) 
This figure starts from the bottom with the vector produced as the output of the decoder stack. It is then turned into an output word. 这个图从底部开始，生成的矢量作为解码器堆栈的输出。 然后它被转换成一个输出单词



## Recap Of Training

## 训练回顾

Now that we’ve covered the entire forward-pass process through a trained Transformer, it would be useful to glance at the intuition of training the model.

既然我们已经通过一个经过训练的 Transformer 涵盖了整个前向传递过程，那么了解一下训练模型的直觉将会非常有用。

During training, an untrained model would go through the exact same forward pass. But since we are training it on a labeled training dataset, we can compare its output with the actual correct output.

在训练过程中，一个未经训练的模特会经历完全相同的向前传球。 但是因为我们是在一个标记的训练数据集上训练它，所以我们可以将它的输出与实际的正确输出进行比较。

To visualize this, let’s assume our output vocabulary only contains six words(“a”, “am”, “i”, “thanks”, “student”, and “<eos>” (short for ‘end of sentence’)).

为了可视化这一点，让我们假设输出词汇表只包含六个单词("a"、"am"、"i"、"thanks"、"student"和"eos"("句子结束"的缩写))。

![img](https://jalammar.github.io/images/t/vocabulary.png)
The output vocabulary of our model is created in the preprocessing phase before we even begin training. 我们模型的输出词汇表是在开始训练之前的预处理阶段创建的

Once we define our output vocabulary, we can use a vector of the same width to indicate each word in our vocabulary. This also known as one-hot encoding. So for example, we can indicate the word “am” using the following vector:

一旦定义了输出词汇表，就可以使用相同宽度的向量来指示词汇表中的每个单词。 这也称为 one-hot 编码。 例如，我们可以用下面的矢量来表示"am"一词:

![img](https://jalammar.github.io/images/t/one-hot-vocabulary-example.png)
Example: one-hot encoding of our output vocabulary 示例: 输出词汇表的 one-hot 编码

Following this recap, let’s discuss the model’s loss function – the metric we are optimizing during the training phase to lead up to a trained and hopefully amazingly accurate model.

在这个回顾之后，让我们讨论一下模型的损失函数——在训练阶段我们正在优化的度量，从而得到一个训练有素并且有希望达到令人惊讶的精确度的模型。

## The Loss Function

## 损失函数

Say we are training our model. Say it’s our first step in the training phase, and we’re training it on a simple example – translating “merci” into “thanks”.

假设我们正在训练我们的模型。 假设这是我们在训练阶段的第一步，我们正在用一个简单的例子来训练它——将"merci"翻译成"thanks"。

What this means, is that we want the output to be a probability distribution indicating the word “thanks”. But since this model is not yet trained, that’s unlikely to happen just yet.

这意味着，我们希望输出是一个表示单词"谢谢"的概率分布。 但是由于这种模式还没有经过训练，这种情况不太可能发生。

![img](https://jalammar.github.io/images/t/transformer_logits_output_and_label.png)
Since the model's parameters (weights) are all initialized randomly, the (untrained) model produces a probability distribution with arbitrary values for each cell/word. We can compare it with the actual output, then tweak all the model's weights using backpropagation to make the output closer to the desired output. 由于模型的参数(权重)都是随机初始化的，(未经训练的)模型会生成一个包含每个单元格 / 单词任意值的概率分布。 我们可以将其与实际输出进行比较，然后使用反向传播法调整所有模型的权重，使输出更接近所需的输出



How do you compare two probability distributions? We simply subtract one from the other. For more details, look at[cross-entropy](https://colah.github.io/posts/2015-09-Visual-Information/) and [Kullback–Leibler divergence](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained).

你如何比较两种概率分布？ 我们只是简单地从一个中减去另一个。 关于更多的细节，看看交叉熵和 Kullback-Leibler 分歧。

But note that this is an oversimplified example. More realistically, we’ll use a sentence longer than one word. For example – input: “je suis étudiant” and expected output: “i am a student”. What this really means, is that we want our model to successively output probability distributions where:

但请注意，这是一个过于简单化的例子。 更现实的做法是，我们使用一个比一个单词长的句子。 例如-input:"je suis tudiant"和预期输出:"i am a student"。 这实际上意味着，我们希望我们的模型是连续输出的概率分布，其中:

- Each probability distribution is represented by a vector of width vocab_size (6 in our toy example, but more realistically a number like 3,000 or 10,000) 每个概率分布由一个宽度单词大小的向量表示(在我们的玩具示例中是6，但更实际的数字是3,000或10,000)
- The first probability distribution has the highest probability at the cell associated with the word “i” 第一个概率分布最有可能出现在与单词"i"相关联的单元格上
- The second probability distribution has the highest probability at the cell associated with the word “am” 第二个概率分布最有可能出现在与单词 am 相关的单元格上
- And so on, until the fifth output distribution indicates ‘ 等等，直到第五个输出分布显示'`<end of sentence>`’ symbol, which also has a cell associated with it from the 10,000 element vocabulary. '符号，它也有一个单元格相关的10,000元素词汇表

![img](https://jalammar.github.io/images/t/output_target_probability_distributions.png)
The targeted probability distributions we'll train our model against in the training example for one sample sentence. 我们将在一个样本句子的训练例子中训练我们的模型对照的目标概率分布



After training the model for enough time on a large enough dataset, we would hope the produced probability distributions would look like this:

在一个足够大的数据集上训练模型足够长的时间后，我们希望生成的概率分布看起来像这样:

![img](https://jalammar.github.io/images/t/output_trained_model_probability_distributions.png)
Hopefully upon training, the model would output the right translation we expect. Of course it's no real indication if this phrase was part of the training dataset (see: 希望经过训练，该模型能够输出我们所期望的正确翻译。 当然，这并不能说明这个短语是否是训练数据集的一部分(参见:[cross validation 交叉验证](https://www.youtube.com/watch?v=TIgfjmp-4BA)). Notice that every position gets a little bit of probability even if it's unlikely to be the output of that time step -- that's a very useful property of softmax which helps the training process. ). 注意，每个位置都有一点概率，即使它不可能是时间步的输出---- 这是 softmax 的一个非常有用的特性，有助于训练过程

Now, because the model produces the outputs one at a time, we can assume that the model is selecting the word with the highest probability from that probability distribution and throwing away the rest. That’s one way to do it (called greedy decoding). Another way to do it would be to hold on to, say, the top two words (say, ‘I’ and ‘a’ for example), then in the next step, run the model twice: once assuming the first output position was the word ‘I’, and another time assuming the first output position was the word ‘me’, and whichever version produced less error considering both positions #1 and #2 is kept. We repeat this for positions #2 and #3…etc. This method is called “beam search”, where in our example, beam_size was two (because we compared the results after calculating the beams for positions #1 and #2), and top_beams is also two (since we kept two words). These are both hyperparameters that you can experiment with.

现在，因为模型一次产生一个输出，我们可以假设模型从概率分布中选择概率最高的单词，然后扔掉其余的。 这是一种方法(称为贪婪解码)。 另一种方法是保持前两个单词(例如,"i"和"a") ，然后在下一步中，运行模型两次: 一次假设第一个输出位置是单词"i"，另一次假设第一个输出位置是单词"me"，考虑到位置 # 1和 # 2，不管哪个版本产生的错误更少，都保持不变。 我们在第2和第3位重复这个动作... ... 等等。 这种方法被称为"梁搜索"，在我们的例子中，梁的大小是2(因为我们在计算了位置 # 1和 # 2的梁之后比较了结果) ，顶梁也是2(因为我们保留了两个单词)。 这两个超参数都可以进行实验。

## Go Forth And Transform

## 勇往直前，改变自己

I hope you’ve found this a useful place to start to break the ice with the major concepts of the Transformer. If you want to go deeper, I’d suggest these next steps:

我希望你已经发现这是一个有用的地方，开始打破冰的主要概念的变压器。 如果你想更进一步，我建议下面的步骤:

- Read the 请阅读[Attention Is All You Need 注意力是你所需要的](https://arxiv.org/abs/1706.03762) paper, the Transformer blog post ( 论文，变压器博客帖子([Transformer: A Novel Neural Network Architecture for Language Understanding 变换器: 一种新的语言理解神经网络结构](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)), and the ) ，以及[Tensor2Tensor announcement 2tensor2tensor 公告](https://ai.googleblog.com/2017/06/accelerating-deep-learning-research.html).
- Watch 看着[Łukasz Kaiser’s talk Ukasz Kaiser 的演讲](https://www.youtube.com/watch?v=rBCqOTEfxvg) walking through the model and its details 穿过模型和它的细节
- Play with the 玩一下[Jupyter Notebook provided as part of the Tensor2Tensor repo 作为 Tensor2Tensor repo 的一部分，Jupyter Notebook 提供](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)
- Explore the 探索[Tensor2Tensor repo 2tensor2tensor repo](https://github.com/tensorflow/tensor2tensor).

Follow-up works:

跟进工作:

- [Depthwise Separable Convolutions for Neural Machine Translation 神经机器翻译中的深度可分卷积](https://arxiv.org/abs/1706.03059)
- [One Model To Learn Them All 一个模式，学会他们所有](https://arxiv.org/abs/1706.05137)
- [Discrete Autoencoders for Sequence Models 序列模型的离散自动编码器](https://arxiv.org/abs/1801.09797)
- [Generating Wikipedia by Summarizing Long Sequences 总结长序列生成维基百科](https://arxiv.org/abs/1801.10198)
- [Image Transformer 图像转换器](https://arxiv.org/abs/1802.05751)
- [Training Tips for the Transformer Model 变压器模型的训练技巧](https://arxiv.org/abs/1804.00247)
- [Self-Attention with Relative Position Representations 自我注意与相对位置表征](https://arxiv.org/abs/1803.02155)
- [Fast Decoding in Sequence Models using Discrete Latent Variables 基于离散潜变量序列模型的快速解码](https://arxiv.org/abs/1803.03382)
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost 副因子: 带次线性存储开销的在线机机器学习](https://arxiv.org/abs/1804.04235)
