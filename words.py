# assumes we take in numpy array [([genres], 'plot')]

import pandas as pd
# read in data (charles code)

raw_data = pd.read_csv('movie_data.csv').values

def transform_data(raw_data):
  genres = raw_data[:,0]
  for i, g in enumerate(genres):
    genres[i] = g.split(':')

  summaries = raw_data[:,1]
  for ind, s in enumerate(summaries): 
    sentences = str(s).split('.')
    for i, sen in enumerate(sentences):
      tokens = sen.split()
      sentences[i] = tokens
    summaries[ind] = sentences
  return genres, summaries