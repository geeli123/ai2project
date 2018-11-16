from keras.layers import LSTM, Input, Dense, Activation
from keras.models import Model
from words import transform_data
import MovieParse

def make_lstm(input_shape):
    shape = input_shape

    inp = Input(shape)

    lay1 = LSTM(10)(inp)

    lay2 = Dense(13)(lay1)

    out = Activation('softmax')(lay2)

    return Model(inp, out)

genre_dict = {'action':0, 'adventure':1, 'drama':2, 'comedy':3,'family':4,
'science fiction':5, 'horror':6, 'animation':7, 'mystery':8, 'thriller':9,
'romance':10, 'crime':11, 'fantasy':12}


def genre_ground_truth(data):
    ground_truth = []
    for genres in data:
        onehot = [0] * 13
        for genre in genres:
            onehot[genre_dict[genre]] = 1
        ground_truth.append(onehot)
    return ground_truth


if __name__ == '__main__':
    raw_data = MovieParse.getData('movie_data.csv')
    print(transform_data(raw_data[0]))
