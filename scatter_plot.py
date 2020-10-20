import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("dataset_train.csv")
    c = df.corr()
    ca = c.abs()
    max1 = 0
    max2 = 0
    for v in c.columns:
        c[v].values[c[v].index.get_loc(v)] = 0
        ca[v].values[ca[v].index.get_loc(v)] = 0
        s = c[v].sort_values()
        sa = ca[v].sort_values()
        if (max1 < s[-1]):
            max1 = s[-1]
            feature_1 = v
            feature_2 = s.index[-1]
        if (max2 < sa[-1]):
            max2 = sa[-1]
            feature_3 = v
            feature_4 = sa.index[-1]
    df = df.set_index("Hogwarts House")
    df1 = df.drop(df.columns.difference([feature_1, feature_2]), axis=1)
    g = df1.drop(df1.index.difference(["Gryffindor"]))
    s = df1.drop(df1.index.difference(["Slytherin"]))
    r = df1.drop(df1.index.difference(["Ravenclaw"]))
    h = df1.drop(df1.index.difference(["Hufflepuff"]))
    plt.figure("Scatter plot 1")
    plt.title("The two similar features")
    plt.scatter(g[feature_1], g[feature_2], c='r', label="Gryffindor", alpha=0.5)
    plt.scatter(s[feature_1], s[feature_2], c='g', label="Slytherin", alpha=0.5)
    plt.scatter(r[feature_1], r[feature_2], c='b', label="Ravenclaw", alpha=0.5)
    plt.scatter(h[feature_1], h[feature_2], c='y', label="Hufflepuff", alpha=0.5)
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.legend()
    df2 = df.drop(df.columns.difference([feature_3, feature_4]), axis=1)
    g = df2.drop(df2.index.difference(["Gryffindor"]))
    s = df2.drop(df2.index.difference(["Slytherin"]))
    r = df2.drop(df2.index.difference(["Ravenclaw"]))
    h = df2.drop(df2.index.difference(["Hufflepuff"]))
    plt.figure("Scatter plot 2")
    plt.title("The two opposite features")
    plt.scatter(g[feature_3], g[feature_4], c='r', label="Gryffindor", alpha=0.5)
    plt.scatter(s[feature_3], s[feature_4], c='g', label="Slytherin", alpha=0.5)
    plt.scatter(r[feature_3], r[feature_4], c='b', label="Ravenclaw", alpha=0.5)
    plt.scatter(h[feature_3], h[feature_4], c='y', label="Hufflepuff", alpha=0.5)
    plt.xlabel(feature_3)
    plt.ylabel(feature_4)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()