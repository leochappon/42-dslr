import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

learning_rate = 0.07
n_iterations = 1000
corr_range = 0.7

def model(X, theta):
    return 1 / (1 + np.exp(-X.dot(theta)))

def cost_function(X, theta, target):
    return np.sum(target * np.log(model(X, theta)) + (1 - target) * np.log(1 - model(X, theta))) / -len(target)

def gradient(X, theta, target):
    return X.T.dot(model(X, theta) - target) / len(target)

def gradient_descent(X, target, learning_rate, n_iterations):
    theta = np.random.randn(X.shape[1], 1)
    cost_history = np.zeros(n_iterations)

    for i in range(n_iterations):
        theta = theta - learning_rate * gradient(X, theta)
        cost_history[i] = cost_function(X, theta, target)
    
    return theta, cost_history

def coef_determination(target, predictions):
	u = np.sum((target - predictions) ** 2)
	v = np.sum((target - np.mean(target)) ** 2)

	return 1 - u / v

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
    hs = df['Hogwarts House']
    df = df._get_numeric_data()
    df = (df - df.mean()) / df.std()
    df = corr_fillna(df)
    df.insert(0, 'Hogwarts House', hs)
    df = df.dropna(axis=0)
    target = df['Hogwarts House']
    features = df.drop('Hogwarts House', axis=1).to_numpy()
    target_g = target.replace({'Gryffindor':1, 'Slytherin':0, 'Ravenclaw':0, 'Hufflepuff':0}).to_numpy().reshape(-1, 1)
    target_s = target.replace({'Gryffindor':0, 'Slytherin':1, 'Ravenclaw':0, 'Hufflepuff':0}).to_numpy().reshape(-1, 1)
    target_r = target.replace({'Gryffindor':0, 'Slytherin':0, 'Ravenclaw':1, 'Hufflepuff':0}).to_numpy().reshape(-1, 1)
    target_h = target.replace({'Gryffindor':0, 'Slytherin':0, 'Ravenclaw':0, 'Hufflepuff':1}).to_numpy().reshape(-1, 1)
    X = np.hstack((features, np.ones((features.shape[0], 1))))

if __name__ == '__main__':
    main()