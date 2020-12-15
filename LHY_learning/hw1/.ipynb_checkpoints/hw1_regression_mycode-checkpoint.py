import sys
import pandas as pd
import numpy as np
# from google.colab import drive

# !gdown --id '1wNKAxQ29G15kgpBy_asjTcZRRgmsCZRm' --output data.zip
# !unzip data.zip
# data = pd.read_csv('gdrive/My Drive/hw1-regression/train.csv', header = None, encoding = 'big5')
data = pd.read_csv('data/hw1_data/train.csv', encoding='big5')
print(data)

data = data.iloc[:, 3:]
print(data)

data[data == "NR"] = 0
raw_data = data.to_numpy()
print(data)
