# naive

NAIVE中文分词器

一个基于tensorflow库的中文开源分词器，

通过人民日报语料训练，准确率97.8%。

除了训练好的分词以外，还带有训练模型，用户可以自己训练其实的数据集。

为和现在其它的中文分词方法给出对比，我们也用该模型训练和测试了sighan bakeoff2005的msr数据。

通过测试脚本给出的正确率达到96.6%。超过了利用神经网络自动提取特征的state of the art.

目前版本还未进行速度方面的优化,我们将在下个版本重写开销大的overhead部分。并且开启自定义词库支持！