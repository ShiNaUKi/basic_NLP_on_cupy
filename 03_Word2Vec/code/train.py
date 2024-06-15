# coding: utf-8
import sys
import numpy as np
sys.path.append('../..')  # 为了引入父目录的文件而进行的设定
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot

import torch.nn.functional as F
import torch

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text) # 1. 转换为数值形式corpus, 对应两份映射字典word_to_id, id_to_word

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])


predict_target =  np.array(corpus[1]).reshape((1, -1))
print(f"predict_target = {id_to_word[predict_target[0].item()]}")
predict_context = np.array([corpus[0], corpus[2]]).reshape((1, -1))
print(f"predict_context = {id_to_word[predict_context[0][0].item()], id_to_word[predict_context[0][1].item()]}")
predict_target = convert_one_hot(predict_target, vocab_size)
predict_context = convert_one_hot(predict_context, vocab_size)
print(f"\n\n{word_to_id.keys()}")
print(f"result is {F.softmax(torch.tensor(model.predict(predict_context), dtype=torch.float32))}")
