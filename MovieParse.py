import pandas as pd
import numpy as np

genres = {'action', 'horror', 'comedy', 'drama'}

def getData(path):
    df = pd.read_csv(path)
    return df[df.Genres.isin(genres)]
