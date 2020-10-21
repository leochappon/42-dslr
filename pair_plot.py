import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pd.read_csv('dataset_train.csv')
    df = df.drop('Index', axis=1)
    palette = {'Gryffindor': 'red', 'Slytherin': 'green', 'Ravenclaw': 'blue', 'Hufflepuff': 'yellow'}
    sns.pairplot(df, hue='Hogwarts House', hue_order=['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff'], palette=palette, diag_kind='hist', dropna=True)
    plt.show()

if __name__ == '__main__':
    main()