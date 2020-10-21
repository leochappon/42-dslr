import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

alpha = 0.07
nb_iterations = 1000
corr_range = 0.7

def corr_fillna(df):
    for v in df.columns:
        for w in df.columns:
            if df[v].corr(df[w]) > corr_range and v != w:
                df[v] = df[v].fillna(df[w])
                df[w] = df[w].fillna(df[v])
            elif df[v].corr(df[w]) < -corr_range and v != w:
                df[v] = df[v].fillna(-df[w])
                df[w] = df[w].fillna(-df[v])
    return df

def main():
    if len(sys.argv) != 2:
        exit('Dataset required')
    df = pd.read_csv(sys.argv[1])
    df = df.drop(['Index', 'Arithmancy', 'Potions', 'Care of Magical Creatures'], axis=1)
    hs = df['Hogwarts House'].map({'Gryffindor':1, 'Slytherin':2, 'Ravenclaw':3, 'Hufflepuff':4})
    df = df._get_numeric_data()
    df = (df-df.mean()) / (df.std())
    df = corr_fillna(df)
    df.insert(0, 'Hogwarts House', hs)
    df = df.dropna(axis=0)
    features = df.drop('Hogwarts House', axis=1).to_numpy()
    target = df['Hogwarts House'].to_numpy().reshape(-1, 1)
    X = np.hstack((features, np.ones((features.shape[0], 1))))

if __name__ == '__main__':
    main()