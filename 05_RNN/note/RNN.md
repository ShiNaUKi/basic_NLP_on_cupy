## RNN的介绍

### 语言建模可写为联合概率P(Word1, Word2, ..., WordN)的判定
1) 基于条件概率进行拆分为P(...) = P(WordN|Word1,..,WordN-1)...P(Word1)的条件概率
2) 基于N阶马尔可夫链的假设, 简化条件概率的建模
3) 若考虑Word2Vec建模，其缺乏对上下文顺便信息的保留, 并且受限于上下文长度


### RNN的引入
1) x0对应的RNN层的输出h0, 与x1一起输入
2) 其中h0成为隐藏状态, 前向传递公式为h_t = tanh(Wh * h_{t-1} + W_x * x_2 + b)
3) 相应的反向传播称为基于时间的反向传播(Backpropagation Through time/ BPTT)
4) 考虑内存资源限制, 将有限个RNN连接, 在有限个传递中学习反向反馈, 而前向反馈不进行截断, 即Truncated RNN