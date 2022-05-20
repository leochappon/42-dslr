import sys
import pandas as pd
import numpy as np
from logreg_train import standardization, corr_fillna, model

def main():
    if (len(sys.argv) != 3):
        exit('Dataset and thetas required')
    df = pd.read_csv(sys.argv[1])
    df = df.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Arithmancy', 'Potions', 'Care of Magical Creatures'], axis=1)
    df = standardization(df)
    df = corr_fillna(df, 0)
    features = df.to_numpy()
    X = np.hstack((features, np.ones((features.shape[0], 1))))
    thetas = pd.read_csv(sys.argv[2])
    theta_g = thetas['Gryffindor']
    theta_s = thetas['Slytherin']
    theta_r = thetas['Ravenclaw']
    theta_h = thetas['Hufflepuff']
    predictions_g = model(X, theta_g)
    predictions_s = model(X, theta_s)
    predictions_r = model(X, theta_r)
    predictions_h = model(X, theta_h)
    houses = ['Gryffindor'] * len(predictions_g)
    for i, v in enumerate(predictions_g):
        p = v
        if p < predictions_s[i]:
            p = predictions_s[i]
            houses[i] = 'Slytherin'
        if p < predictions_r[i]:
            p = predictions_r[i]
            houses[i] = 'Ravenclaw'
        if p < predictions_h[i]:
            p = predictions_h[i]
            houses[i] = 'Hufflepuff'
    houses = pd.DataFrame(houses, columns=['Hogwarts House'])
    houses = houses.rename_axis('Index')
    houses.to_csv('houses.csv')

if __name__ == '__main__':
    main()