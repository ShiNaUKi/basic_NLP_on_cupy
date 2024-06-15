# coding: utf-8
import sys
sys.path.append('../..')
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb


corpus, word_to_id, id_to_word = ptb.load_data('train')

# print('corpus size:', len(corpus))
# print('corpus[:30]:', corpus[:30])
# print()
# print('id_to_word[0]:', id_to_word[0])
# print('id_to_word[1]:', id_to_word[1])
# print('id_to_word[2]:', id_to_word[2])
# print()
# print("word_to_id['car']:", word_to_id['car'])
# print("word_to_id['happy']:", word_to_id['happy'])
# print("word_to_id['lexus']:", word_to_id['lexus'])


# def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
# create_co_matrix(corpus, vocab_size, window_size=1)
# ppmi

keyword = 'car'
C = create_co_matrix(corpus, len(id_to_word.keys()))
PPMI_matrix = ppmi(C)

# 比较两类方法的区别
print("\n\n based on the Co-ocurrance matrix")
most_similar(keyword, word_to_id, id_to_word, C)
print("\n\n based on the PPMI matrix")
most_similar(keyword, word_to_id, id_to_word, PPMI_matrix)

# 降维方法
wordvec_size = 100
print('calculating SVD ...')
try:
    # truncated SVD (fast!)
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(PPMI_matrix, n_components=wordvec_size, n_iter=5,
                             random_state=None)
except ImportError:
    # SVD (slow)
    U, S, V = np.linalg.svd(PPMI_matrix)

print("\n\n based on the SVD algorithm processed on the PPMI matrix")
most_similar(keyword, word_to_id, id_to_word, U)

# based on the Co-ocurrance matrix
#
# [query] car
#  winner: 0.7496030685385788
#  statement: 0.7464830059412864
#  day: 0.7450142511719002
#  computer: 0.7364509377067971
#  character: 0.7326343259148129
#
#
#  based on the PPMI matrix
#
# [query] car
#  auto: 0.20445512235164642
#  cars: 0.1365518569946289
#  disk-drive: 0.12632137537002563
#  personal-computer: 0.1258602738380432
#  truck: 0.12358210980892181
# calculating SVD ...
#
#
#  based on the SVD algorithm processed on the PPMI matrix
#
# [query] car
#  auto: 0.7219744324684143
#  luxury-car: 0.6084914803504944
#  truck: 0.5852997303009033
#  chemical: 0.5770130753517151
#  semiconductor: 0.5538449883460999