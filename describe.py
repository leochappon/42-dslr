import numpy as np
import pandas as pd
import sys

def ft_count(column):
    return len(column)

def ft_mean(column):
    total = 0
    for v in column:
        total += v
    return total / len(column)

def ft_std(column):
    total = 0
    mean = ft_mean(column)
    for v in column:
        total += (mean - v) ** 2
    return (total / (len(column) - 1)) ** 0.5

def ft_min(column):
    return np.sort(column)[0]

def ft_percentile(column, percentage):
    column = np.sort(column)
    p = (len(column) - 1) * (percentage / 100)
    if p.is_integer():
        return column[int(p)]
    f = np.floor(p)
    c = np.ceil(p)
    d0 = column[int(f)] * (c - p)
    d1 = column[int(c)] * (p - f)
    return d0 + d1

def ft_max(column):
    return np.sort(column)[-1]

def describe(df):
    df = df._get_numeric_data()
    columns = df.columns.to_numpy()
    column = []
    for v in columns:
        column.append(df[v].dropna().to_numpy())
    count, mean, std, min, p25, p50, p75, max = ([] for _ in range(8))
    for v in column:
        if v.size == 0:
            count.append(0)
            mean.append("NaN")
            std.append("NaN")
            min.append("NaN")
            p25.append("NaN")
            p50.append("NaN")
            p75.append("NaN")
            max.append("NaN")
        else:
            count.append(ft_count(v))
            mean.append(ft_mean(v))
            std.append(ft_std(v))
            min.append(ft_min(v))
            p25.append(ft_percentile(v, 25))
            p50.append(ft_percentile(v, 50))
            p75.append(ft_percentile(v, 75))
            max.append(ft_max(v))
    data = [count, mean, std, min, p25, p50, p75, max]
    index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    return pd.DataFrame(data, index, columns)

def main():
    if len(sys.argv) != 2:
        sys.exit('Dataset required')
    df = pd.read_csv(sys.argv[1])
    df = describe(df)
    print(df)

if __name__ == '__main__':
    main()