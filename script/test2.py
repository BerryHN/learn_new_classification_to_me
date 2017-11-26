# -*- coding: utf-8 -*-
# author:BerryHN
import jieba
import random


from sklearn.datasets  import fetch_20newsgroups

categories = ['comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x']

newsgroup_train = fetch_20newsgroups(subset = 'train',categories = categories)


from pprint  import pprint
pprint(list(newsgroup_train.target_names))

from sklearn.feature_extraction.text import CountVectorizer


documents=['The dog ate a sandwich, the wizard transfigured a sandwich, and I ate a sandwich']
vector=CountVectorizer(stop_words='english')

print (vector)
print(vector.fit_transform(documents).todense())
print(vector.vocabulary_)

from sklearn.feature_extraction.text import HashingVectorizer
documents=['The dog ate a sandwich and I ate a sandwich','The wizard transfigured a sandwich']
vector=HashingVectorizer(n_features=5)
print(vector.transform(documents).todense())


from sklearn.feature_extraction import DictVectorizer
onehot_encoder = DictVectorizer()
instances = [{'city':'New York'},{'city':'San Francisco'},{'city': 'Chapel Hill'}]
print (onehot_encoder.fit_transform(instances).toarray())




from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
vectorizer = CountVectorizer()
print (vectorizer.fit_transform(corpus).todense())
print (vectorizer.vocabulary_)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(corpus).todense()
for x,y in [[0,1],[0,2],[1,2]]:
    dist = euclidean_distances(counts[x],counts[y])
    print('文档{}与文档{}的距离{}'.format(x,y,dist))


from sklearn.feature_extraction.text import  CountVectorizer
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
vectorizer = CountVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('gathering','v'))
print(lemmatizer.lemmatize('gathering','n'))


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
'The dog ate a sandwich and I ate a sandwich',
'The wizard transfigured a sandwich'
]
vectorizer = TfidfVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

from sklearn.feature_extraction.text import HashingVectorizer
corpus = ['the', 'ate', 'bacon', 'cat']
vectorizer = HashingVectorizer(n_features=6)
print(vectorizer.transform(corpus).todense())


import  nltk.classify.NaiveBayesClassifier

import sklearn.naive_bayes
sklearn.naive_bayes.MultinomialNB.predict()
