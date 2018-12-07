from keras.layers import LSTM, Input, Dense, Activation, Bidirectional
from keras.models import Model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from gensim.models import Word2Vec
from words import *
import MovieParse

def make_lstm(input_shape):
    shape = input_shape

    inp = Input(shape)

    lay1 = Bidirectional(LSTM(10))(inp)

    lay2 = Dense(4)(lay1)

    out = Activation('softmax')(lay2)
    
    model = Model(inp, out)

    return model

#genre_dict = {'action':0, 'adventure':1, 'drama':2, 'comedy':3,'family':4,
#science fiction':5, 'horror':6, 'animation':7, 'mystery':8, 'thriller':9,
#'romance':10, 'crime':11, 'fantasy':12}
genre_dict = {'action':0, 'horror':1, 'comedy':2, 'drama':3}
genre_dict_reverse = {v:k for k,v in genre_dict.items()}


def genre_ground_truth(data):
    ground_truth = []
    for genres in data:
        onehot = [0] * len(genre_dict)
        for genre in genres:
            onehot[genre_dict[genre]] = 1
        ground_truth.append(onehot)
    return np.array(ground_truth)

def create_data_and_save(genre_file_name, words_file_name):
    df = MovieParse.getData('movie_data.csv')

    num_samples = 945
    action = MovieParse.getSample(df, 'action', num_samples)
    horror = MovieParse.getSample(df, 'horror', num_samples)
    comedy = MovieParse.getSample(df, 'comedy', num_samples)
    drama = MovieParse.getSample(df, 'drama', num_samples)

    raw_data = np.vstack((action,horror,comedy,drama))
    print(raw_data.shape)
    np.random.shuffle(raw_data)

    genres, summaries = transform_data_words(raw_data)

    model = Word2Vec.load('word2vec.model')

    """
    max = 0
    for summ in summaries:
        if max < len(summ):
            max = len(summ)

    min = 1425
    for summ in summaries:
        if min > len(summ):
            min = len(summ)

    dictionary = {}
    for summ in summaries:
        length = len(summ) // 100
        dictionary[length] = dictionary.get(length,0) + 1

    print(min)
    print(dictionary)
    """

    word_limit = 75
    input_data = np.zeros((len(summaries),word_limit, 25))
    #input_data = np.zeros((len(summaries),max, 25))

    for ind1,summ in enumerate(summaries):
        for ind2,word in enumerate(summ):
            if ind2 < word_limit:
                input_data[ind1,ind2] = model.wv[word]
        
    ground_truth = genre_ground_truth(genres)

    np.save(genre_file_name, ground_truth)
    np.save(words_file_name, input_data)


if __name__ == '__main__':

    genre_file_name = 'data/genre_data.npy'
    words_file_name = 'data/input_data.npy'

    create_data_and_save(genre_file_name, words_file_name)

    ground_truth = np.load(genre_file_name)
    input_data = np.load(words_file_name)

    print(input_data)
    print(ground_truth)
    print(input_data.shape)
    print(ground_truth.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(input_data, ground_truth, stratify=ground_truth, test_size=0.2)
    lstm_model = make_lstm(X_train.shape[1:])

    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train, Y_train, epochs=50, batch_size=100)

    predictions = lstm_model.predict(X_test)

    preds = np.asarray([np.argmax(line) for line in predictions])
    best_preds = np.zeros(predictions.shape)
    for ind,pred in enumerate(preds):
        best_preds[ind][pred] = 1

    #print(predictions)
    #print(preds)
    #print(best_preds)
    print(Y_test)
    print(best_preds)
    print(Y_test.shape)
    print(best_preds.shape)

    print('-----')
    print('precision: ' + str(precision_score(Y_test, best_preds, average='macro')))
    print('recall: ' + str(recall_score(Y_test, best_preds, average='macro')))
    print('f1 score: : ' + str(f1_score(Y_test, best_preds, average='macro')))
