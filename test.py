import os
from torchtext import data

path = 'train_data.csv'
examples = []
with open(path, errors='ignore') as f:
    f = list(f)
    f = f[:10]
    for line in f:

        print(line[:-3])
        print(line[-2])

