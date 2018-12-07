# assumes we take in numpy array [([genres], 'plot')]

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from pattern.en import singularize, lemma
from sklearn.model_selection import train_test_split
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

mapping = {'action':1, 'horror':2, 'comedy':3, 'drama':0}
genres = map(lambda x:mapping[x], genres)

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

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test ,label=y_test)

for depth in [3, 5, 7]:
  for eta in [0.05, 0.1, 0.2, 0.3]:
    for min_child_weight in [1, 3, 5]:
      param = {
        'max_depth': depth,
        'min_child_weight':min_child_weight,
        'eta': eta,
        'silent': 1, 
        'objective': 'multi:softprob',
        'num_class': 4
      }

      bst = xgb.train(param, dtrain, 20)

      predictions = bst.predict(dtest)
      best_preds = np.asarray([np.argmax(line) for line in predictions])

      print '-----'
      print 'max depth: ' + str(depth) + ', eta: ' + str(eta) + ', min_child_weight: ' + str(min_child_weight)
      print 'precision: ' + str(precision_score(y_test, best_preds, average='macro'))
      print 'recall: ' + str(recall_score(y_test, best_preds, average='macro'))
      print 'f1 score: : ' + str(f1_score(y_test, best_preds, average='macro'))


