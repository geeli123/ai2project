# assumes we take in numpy array [([genres], 'plot')]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from pattern.en import singularize, lemma
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')

# convert word to base form (no possessive, base form if verb, singular noun)
# all lower case, strip of unnecessary characters
def normalizeWord(word):
  purified_word = word.lower().strip('<>:,-";().[]') # remove extraneous chars
  return singularize(lemma(purified_word))  # return base form of word 

# read in data (charles code)
raw_data = pd.read_csv('movie_data.csv').values

genres = raw_data[:,0]
for i, g in enumerate(genres):
  genres[i] = g.split(':')
summaries = raw_data[:,1]
for ind, s in enumerate(summaries): 
  tokens = str(s).split()
  for i, v in enumerate(tokens):
    tokens[i] = normalizeWord(v)
  words = [word for word in tokens if word.isalpha()]
  summaries[ind] = words

tfidf_vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'))
summary_vectors = tfidf_vectorizer.fit_transform(summaries)
print summary_vectors[0]

info = pd.DataFrame(zip(genres, summary_vectors))
info.to_csv('embeddings.csv', encoding='utf-8', index=False)
'''
# convert genre labels to sparse matrix format for processing
mlb = MultiLabelBinarizer()
sparse_genres = mlb.fit_transform(genres)
genre_labels = mlb.classes_

# TODO - get summary_vector statistics and actually do classification
clf = MultinomialNB()
'''
