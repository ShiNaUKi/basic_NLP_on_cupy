## 自然语言的分布式表示

### 主要三种方式: 同义词字典、统计特征及Word2vec

### 同义词字典
1) 最典型案例: WordNet
```
pip install nltk
from nltk.corpus import wordnet

# 查看'car'的同义词
wordnet.synsets('car')

# 查看某个同义簇
car = wordnet.synsets('car.n.01')
car.definition()

# 查看同义簇中的所有近义词
car.lemma_names()

# 查看指定词语的上位词 (即上方的树型结构)
car.hypernym_paths()[0]

# 基于上位词结构计算相似度
dog = wordnet.synsets('dog.n.01')
novel = wordnet.synsets('novel.n.01')
motorcycle = wordnet.synset('motorcycle.n.01')
car.path_similarity(dog)
car.path_similarity(novel)
car.path_similarity(motorcycle)
```

所存在主要问题:
1) 难以适应语义变化
2) 人力成本高
3) 难以表示单词间微妙差异

### 统计特征——基于计数的方法
1) preprocess 将句子词语转化为数值序列
2) 分布假设(单词含义由上下文决定)建模词语 -> 计数统计 -> 共现矩阵计算(Co-occurance Matrix)
3) 提取高频无意词的影响 -> PPMI -> 正点互(Pointwise Mutual Information Matrix)矩阵
4) 降低维度 -> SVD算法(奇异值分解)
5) 常见PTB数据集

