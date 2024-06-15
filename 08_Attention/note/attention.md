## Attention的应用

### Seq2Seq所存在问题
1) 编码器所生成结果, 其大小十分固定
2) 将编码器产生的所有潜在特征都输入到解码器中
3) 并且利用Attention结构进行相关性评估


### Attention的其他应用
1) 双向RNN
2) 考虑多层LSTM存在的Skip Connection
3) GNMT (Google Nerual Machine Transaction)
4) Transformer (自注意力机制 / 实现并行的序列预测)
5) NTM(Nerual Tuning Machine) DeepMind()