# assumes we take in numpy array [([genres], 'plot')]

from sklearn.feature_extraction.text impomrt TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from pattern.en import singularize, lemma
import numpy as np

# convert word to base form (no possessive, base form if verb, singular noun)
# all lower case, strip of unnecessary characters
def normalizeWord(word):
  purified_word = word.strip('<>:,-";().[]').replace('&lt;','').replace('&gt;','') # remove extraneous chars
  non_possessed_word = purified_word[:-2] if purified_word[-2:] == "'s" else purified_word # make word non-possessive
  return singularize(lemma(non_possessed_word))  # return base form of word 

# TODO - read in data (charles code)
# raw_data = 

genres = raw_data[:,0]
summaries = raw_data[:,1]

tfidf_vectorizer = TfidfVectorizer()
summary_vectors = tfidf_vectorizer.fit_transform(summaries)

# convert genre labels to sparse matrix format for processing
mlb = MultiLabelBinarizer()
sparse_genres = mlb.fit_transform(genres)
genre_labels = mlb.classes_

# TODO - get summary_vector statistics and actually do classification
clf = MultinomialNB()
