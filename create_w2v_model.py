import numpy as np
from gensim.models import Word2Vec
import pandas as pd 
from words import transform_data_words


raw_data = pd.read_csv('movie_data.csv').values

genres, summaries =  transform_data_words(raw_data)

model = Word2Vec(summaries, size=25, window=5, workers=6, min_count=0)

model.save('word2vec.model')

print(model.wv['in'])
print(model.wv['in'].shape)