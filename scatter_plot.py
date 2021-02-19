import pandas as pd
import matplotlib.pyplot as plt

def scatter_plot(df, feature_x, feature_y, string):
    df = df.set_index("Hogwarts House")
    df = df.drop(df.columns.difference([feature_x, feature_y]), axis=1)
    g = df.drop(df.index.difference(["Gryffindor"]))
    s = df.drop(df.index.difference(["Slytherin"]))
    r = df.drop(df.index.difference(["Ravenclaw"]))
    h = df.drop(df.index.difference(["Hufflepuff"]))
    plt.figure()
    plt.title(string)
    plt.scatter(g[feature_x], g[feature_y], c='r', label="Gryffindor", alpha=0.5)
    plt.scatter(s[feature_x], s[feature_y], c='g', label="Slytherin", alpha=0.5)
    plt.scatter(r[feature_x], r[feature_y], c='b', label="Ravenclaw", alpha=0.5)
    plt.scatter(h[feature_x], h[feature_y], c='y', label="Hufflepuff", alpha=0.5)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()

def main():
    df = pd.read_csv("dataset_train.csv")
    c = df.corr()
    max = 0
    min = 0
    for v in c.columns:
        c[v].values[c[v].index.get_loc(v)] = 0
        s = c[v].sort_values()
        if max <= s[-1]:
            max = s[-1]
            feature_1 = v
            feature_2 = s.index[-1]
        if min >= s[0]:
            min = s[0]
            feature_3 = v
            feature_4 = s.index[0]
    scatter_plot(df, feature_1, feature_2, "The 2 most similar features")
    scatter_plot(df, feature_3, feature_4, "The 2 most different features")
    plt.show()

if __name__ == '__main__':
    main()