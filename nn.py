# assumes we take in numpy array [([genres], 'plot')]

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from pattern.en import singularize, lemma
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
import xgboost as xgb
import numpy as np
import pandas as pd
import nltk


genres = {'action', 'horror', 'comedy', 'drama'}

def getSample(df, genre, k):
  return df[df.Genres.isin([genre])].sample(n=k)

# convert word to base form (no possessive, base form if verb, singular noun)
# all lower case, strip of unnecessary characters
def normalizeWord(word):
  purified_word = word.lower().strip('<>:,-";().[]') # remove extraneous chars
  return singularize(lemma(purified_word))  # return base form of word 

# read in data (charles code)
df = pd.read_csv('movie_data.csv')
raw_data = df[df.Genres.isin(genres)]

act = getSample(raw_data, 'action', 940)
horror = getSample(raw_data, 'horror', 940) 
comedy = getSample(raw_data, 'comedy', 940) 
drama = getSample(raw_data, 'drama', 940) 

raw_data = pd.concat([act, horror, comedy, drama]).values

genres = raw_data[:,0]
mapping = {'action':(0,0,0,1), 'horror':(0,0,1,0), 'comedy':(0,1,0,0), 'drama':(1,0,0,0)}
mapping2 = {3:(0,0,0,1), 2:(0,0,1,0), 1:(0,1,0,0), 0:(1,0,0,0)}
genres = np.asarray(map(lambda x:mapping[x], genres))

summaries = raw_data[:,1]

for ind, s in enumerate(summaries): 
  tokens = str(s).split()
  for i, v in enumerate(tokens):
    tokens[i] = normalizeWord(v)
  words = [word for word in tokens if word.isalpha()]
  summaries[ind] = ' '.join(words)
tfidf_vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'))
summary_vectors = tfidf_vectorizer.fit_transform(summaries)

X_train, X_test, y_train, y_test = train_test_split(summary_vectors, genres, stratify=genres, test_size=0.2)
print summary_vectors[0].shape
model = Sequential()
model.add(Dense(300, activation='relu', input_dim=summary_vectors[0].shape[1]))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=128)

predictions = model.predict(X_test)
best_preds = np.asarray([np.argmax(line) for line in predictions])
best_preds = np.asarray(map(lambda x: mapping2[x], best_preds)) 
print '-----'
print 'precision: ' + str(precision_score(y_test, best_preds, average='macro'))
print 'recall: ' + str(recall_score(y_test, best_preds, average='macro'))
print 'f1 score: : ' + str(f1_score(y_test, best_preds, average='macro'))

