# Machine Learning Engineer Nanodegree
## Capstone Proposal
Yuanchao Ma

March 20, 2019
## Proposal
### Domain Background
互联网的本质是为了降低人们获取信息的成本，更便捷的进行沟通和分享。因此，从互联网创建时，就允许世界各地的人们通过互联网进行自由的交流、讨论以及合作。而像国内的贴吧，微博，微信，国外的Twitter，Facebook，Wikipedia等社区平台的建立，形成了这些互动可以发生的基础。为了人们在社区中可以更有序的交流以及促进对话，许多的社区都制定了自己的标准和规则，并防止这些社区被有毒行为劫持或摧毁。然而，随着有毒评论的黑色产业化，利益驱使人们通过各种手段来规避规范和标准，使得通过人为来执行这些规范和标准变得越来越困难。事实上Facebook正在招聘越来越多的版主来筛选可疑的内容[1]。同时，许多新闻网站现在也已经开始禁用评论功能[2]。而这些人工的审核监控机制，是非常低效的做法。

综上，我们需要一种工具来自动化地对用户评论进行监视，分类和标记。此外，不同的网站可能需要监控不同类型的内容。因此需要建立一个能够区分不同类型的言语攻击行为的模型。

我们可以看到在论文[3]中，研究人员对情感分析进行了大量研究。他们的工作重点是情绪分析，这与我们正在研究的领域非常相似。论文中定义了一种使用词袋技术预处理文本的合理方法。他们接着使用SVM和朴素贝叶斯分类器来确定推文的情绪是积极的，中性的还是负面的，并且发现朴素贝叶斯分类器更准确。此外，当他们对推文进行矢量化时，他们通过使用bigrams来提高分类器的准确性。他们的工作可以为我的benchmark model参考。

### Problem Statement
Toxic Comment Classification Challenge是kaggle上由Jigsaw提出的一个比赛，旨在找到更好的对恶毒评论的多分类模型。我通过参加这个比赛，使用比赛提供的人工标注的Wikipedia评论数据，训练一个能够在任意文本数据上判断多种恶意（威胁，色情，侮辱和种族歧视言论等）分类概率的多分类模型。

### Datasets and Inputs
训练数据由Toxic Comment Classification Challenge比赛提供。数据为对恶性行为人工标注的Wikipedia评论数据。标注的类型为：
* toxic
* severe_toxic
* obscene
* threat
* insult
* indentity_hate
比赛提供的数据由如下四个文件构成：
* train.csv - 
### Solution Statement
### Benchmark Model
### Evaluation Metircs
### Project Design

### Reference
1. http://fortune.com/2018/03/22/human-moderators-facebook-youtube-twitter/
2. https://www.theguardian.com/science/brain-flapping/2014/sep/12/comment-sections-toxic-moderation
3. http://crowdsourcing-class.org/assignments/downloads/pak-paroubek.pdf
4. 
