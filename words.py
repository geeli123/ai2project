# assumes we take in numpy array [([genres], 'plot')]

import pandas as pd
# read in data (charles code)

#raw_data = pd.read_csv('movie_data.csv').values
def transform_data_genres(raw_data):
    genres = raw_data
    for i, g in enumerate(genres):
      genres[i] = g.split(':')

    return genres

def transform_data_sentences(raw_data):
    genres = transform_data_genres(raw_data[:,0])

    summaries = raw_data[:,1]
    for ind, s in enumerate(summaries): 
      sentences = str(s).split('.')
      for i, sen in enumerate(sentences):
        tokens = sen.split()
        sentences[i] = tokens
      summaries[ind] = sentences
    return genres, summaries


def transform_data_words(raw_data):
    genres = transform_data_genres(raw_data[:,0])

    summaries = raw_data[:,1]
    ret_summaries = []
    for s in summaries: 
        tokens = s.split()
        document = []
        for token in tokens:
            stripped = token.strip('-,.!~<>;?')
            lower = stripped.lower()
            if lower.isalpha():
              document.append(lower)
        ret_summaries.append(document)
    return genres, ret_summaries