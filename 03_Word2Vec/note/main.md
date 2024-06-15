## Word2Vec的两种方式

### 1. CBOW (Continous Bag-of-Words)
1) 基于目标词语的上下文进行预测
2) 上下文输入共享权重
3) 多数采用W_in代替分布信息


### 2. Skip-gram 
1) 基于目标词语预测上下文
2) 两种中间输出共享参数权重