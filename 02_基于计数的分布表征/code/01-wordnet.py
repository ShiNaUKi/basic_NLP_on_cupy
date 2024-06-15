# pip install nltk
from nltk.corpus import wordnet

# 查看'car'的同义词
print(f"synsets = {wordnet.synsets('food')}")

# 查看某个同义簇
food = wordnet.synset('food.n.01')
print(f"car_defi = {food.definition()}")

# 查看同义簇中的所有近义词
print(f"car_lemma_names = {food.lemma_names()}")


# 查看指定词语的上位词 (即上方的树型结构)
print(f"car_hypernym_paths = {food.hypernym_paths()[0]}")

# 基于上位词结构计算相似度
dog = wordnet.synset('dog.n.01')
novel = wordnet.synset('novel.n.01')
motorcycle = wordnet.synset('motorcycle.n.01')
banana = wordnet.synset('banana.n.01')

# food.path_similarity(dog)
# food.path_similarity(novel)
# food.path_similarity(motorcycle)

print(f"path_similarity of food and dog = {food.path_similarity(dog)}")
print(f"path_similarity of food and novel = {food.path_similarity(novel)}")
print(f"path_similarity of food and motorcycle = {food.path_similarity(motorcycle)}")
print(f"path_similarity of food and banana = {food.path_similarity(banana)}")