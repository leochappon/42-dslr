import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

learning_rate = 0.01
n_iterations = 2000

def standardization(df):
    return (df - df.mean()) / df.std()

def corr_fillna(df, corr):
    new_df = df
    i = 1

    while i >= corr and new_df.isnull().values.any():
        for v in df.columns:
            for w in df.columns:
                if df[v].corr(df[w]) >= i and v != w:
                    new_df[v] = new_df[v].fillna(df[w])
                    new_df[w] = new_df[w].fillna(df[v])
                elif df[v].corr(df[w]) <= -i and v != w:
                    new_df[v] = new_df[v].fillna(-df[w])
                    new_df[w] = new_df[w].fillna(-df[v])
        i -= 0.1

    return new_df

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

def sct_target_predictions(classes, features, target, predictions, house, color):
    for i in range(len(features)):
        plt.figure(f'{house} - {classes[i]}')
        plt.title(classes[i])
        plt.scatter(features[i], predictions, c=color, alpha=0.5)
        plt.scatter(features[i], target, alpha=0.5)
        plt.xlabel(f'Grades in {classes[i]}')
        plt.ylabel(f'Probability of being in {house}')
    plt.show()

def plt_cost_history(cost_history, house, c):
    plt.xlabel('n_iterations')
    plt.ylabel('cost_history')
    plt.plot(range(n_iterations), cost_history, color=c, label=house)

def main():
    if len(sys.argv) != 2:
        exit('Dataset required')
    df = pd.read_csv(sys.argv[1])
    df = df.drop(['Index', 'Arithmancy', 'Potions', 'Care of Magical Creatures'], axis=1)
    hs = df['Hogwarts House']
    df = df._get_numeric_data()
    df = standardization(df)
    df = corr_fillna(df, 0.7)
    df.insert(0, 'Hogwarts House', hs)
    df = df.dropna()

    classes = df.drop('Hogwarts House', axis=1)
    features = classes.to_numpy()
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
    np.savetxt('thetas.csv', thetas, delimiter=',', header='Gryffindor,Slytherin,Ravenclaw,Hufflepuff', comments='')

    predictions_g = model(X, theta_g)
    predictions_s = model(X, theta_s)
    predictions_r = model(X, theta_r)
    predictions_h = model(X, theta_h)

    features = np.transpose(features)
    sct_target_predictions(classes.columns, features, target_g, predictions_g, 'Gryffindor', 'r')
    sct_target_predictions(classes.columns, features, target_s, predictions_s, 'Slytherin', 'g')
    sct_target_predictions(classes.columns, features, target_r, predictions_r, 'Ravenclaw', 'b')
    sct_target_predictions(classes.columns, features, target_h, predictions_h, 'Hufflepuff', 'y')

    plt.figure()
    plt.title('Cost history of the Hogwarts houses')
    plt_cost_history(cost_history_g, "Gryffindor", 'r')
    plt_cost_history(cost_history_s, "Slytherin", 'g')
    plt_cost_history(cost_history_r, "Ravenclaw", 'b')
    plt_cost_history(cost_history_h, "Hufflepuff", 'y')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()