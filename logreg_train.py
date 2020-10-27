import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

learning_rate = 0.01
n_iterations = 2000
corr_range = 0.7

def corr_fillna(df):
    for v in df.columns:
        for w in df.columns:
            if df[v].corr(df[w]) >= corr_range and v != w:
                df[v] = df[v].fillna(df[w])
                df[w] = df[w].fillna(df[v])
            elif df[v].corr(df[w]) <= -corr_range and v != w:
                df[v] = df[v].fillna(-df[w])
                df[w] = df[w].fillna(-df[v])
    return df

def model(X, theta):
    return 1 / (1 + np.exp(-X.dot(theta)))

def cost_function(X, theta, target):
    return np.sum(target * np.log(model(X, theta)) + (1 - target) * np.log(1 - model(X, theta))) / -len(target)

def gradient(X, theta, target):
    return X.T.dot(model(X, theta) - target) / len(target)

def gradient_descent(X, target):
    theta = np.random.randn(X.shape[1], 1)
    cost_history = np.zeros(n_iterations)

    for i in range(n_iterations):
        theta = theta - learning_rate * gradient(X, theta, target)
        cost_history[i] = cost_function(X, theta, target)

    return theta, cost_history

def coef_determination(target, predictions):
	u = np.sum((target - predictions) ** 2)
	v = np.sum((target - np.mean(target)) ** 2)

	return 1 - u / v

def plt_cost_history(cost_history, house, c):
    plt.xlabel('n_iterations')
    plt.ylabel('cost_history')
    plt.plot(range(n_iterations), cost_history, c=c, label=house)

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

    features = df.drop('Hogwarts House', axis=1).to_numpy()
    X = np.hstack((features, np.ones((features.shape[0], 1))))

    target = df['Hogwarts House']
    target_g = target.replace({'Gryffindor':1, 'Slytherin':0, 'Ravenclaw':0, 'Hufflepuff':0}).to_numpy().reshape(-1, 1)
    target_s = target.replace({'Gryffindor':0, 'Slytherin':1, 'Ravenclaw':0, 'Hufflepuff':0}).to_numpy().reshape(-1, 1)
    target_r = target.replace({'Gryffindor':0, 'Slytherin':0, 'Ravenclaw':1, 'Hufflepuff':0}).to_numpy().reshape(-1, 1)
    target_h = target.replace({'Gryffindor':0, 'Slytherin':0, 'Ravenclaw':0, 'Hufflepuff':1}).to_numpy().reshape(-1, 1)

    theta_g, cost_history_g = gradient_descent(X, target_g)
    theta_s, cost_history_s = gradient_descent(X, target_s)
    theta_r, cost_history_r = gradient_descent(X, target_r)
    theta_h, cost_history_h = gradient_descent(X, target_h)

    thetas = np.column_stack((theta_g, theta_s, theta_r, theta_h))
    np.savetxt('thetas.csv', thetas, delimiter=',', header='Gryffindor, Slytherin, Ravenclaw, Hufflepuff', comments='')

    predictions_g = model(X, theta_g)
    predictions_s = model(X, theta_s)
    predictions_r = model(X, theta_r)
    predictions_h = model(X, theta_h)
    print("Coefficient of determination for Gryffindor: {}".format(coef_determination(target_g, predictions_g)))
    print("Coefficient of determination for Slytherin: {}".format(coef_determination(target_s, predictions_s)))
    print("Coefficient of determination for Ravenclaw: {}".format(coef_determination(target_r, predictions_r)))
    print("Coefficient of determination for Hufflepuff: {}".format(coef_determination(target_h, predictions_h)))

    plt.figure()
    plt.title('Cost history')
    plt_cost_history(cost_history_g, "Gryffindor", 'r')
    plt_cost_history(cost_history_s, "Slytherin", 'g')
    plt_cost_history(cost_history_r, "Ravenclaw", 'b')
    plt_cost_history(cost_history_h, "Hufflepuff", 'y')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()