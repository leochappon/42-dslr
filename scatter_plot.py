import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("dataset_train.csv")
    c = df.corr()
    max = 0
    for v in c.columns:
        c[v].values[c[v].index.get_loc(v)] = 0
        s = c[v].sort_values()
        if (max < s[-1]):
            max = s[-1]
            feature_1 = v
            feature_2 = s.index[-1]
    df = df.set_index("Hogwarts House")
    df = df.drop(df.columns.difference([feature_1, feature_2]), axis=1)
    g = df.drop(df.index.difference(["Gryffindor"]))
    s = df.drop(df.index.difference(["Slytherin"]))
    r = df.drop(df.index.difference(["Ravenclaw"]))
    h = df.drop(df.index.difference(["Hufflepuff"]))
    plt.figure("Scatter plot")
    plt.title("The two similar features")
    plt.scatter(g[feature_1], g[feature_2], c='r', label="Gryffindor", alpha=0.5)
    plt.scatter(s[feature_1], s[feature_2], c='g', label="Slytherin", alpha=0.5)
    plt.scatter(r[feature_1], r[feature_2], c='b', label="Ravenclaw", alpha=0.5)
    plt.scatter(h[feature_1], h[feature_2], c='y', label="Hufflepuff", alpha=0.5)
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()