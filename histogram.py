import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('dataset_train.csv')
    df = df.set_index('Hogwarts House')
    df = df._get_numeric_data()
    df = df.drop('Index', axis=1)
    g = df.drop(df.index.difference(['Gryffindor']))
    s = df.drop(df.index.difference(['Slytherin']))
    r = df.drop(df.index.difference(['Ravenclaw']))
    h = df.drop(df.index.difference(['Hufflepuff']))
    for v in df.columns:
        plt.figure()
        plt.title(v)
        plt.hist(g[v], color='r', label='Gryffindor', alpha=0.5)
        plt.hist(s[v], color='g', label='Slytherin', alpha=0.5)
        plt.hist(r[v], color='b', label='Ravenclaw', alpha=0.5)
        plt.hist(h[v], color='y', label='Hufflepuff', alpha=0.5)
        plt.ylabel('Frequency')
        plt.xlabel('Grades')
        plt.legend(frameon=False)
    plt.show()

if __name__ == '__main__':
    main()