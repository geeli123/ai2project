import pandas as pd

def getData(path):
    return pd.read_csv(path).values
