import numpy as np
import pandas as pd
# This is the library for the Reservoir Computing got it by: https://github.com/cknd/pyESN
from pyESN import ESN
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

df = pd.read_csv('train.csv', sep=',')

cols = ["Close/Last", "Open", "High", "Low", "NextOpen"]
df[cols] = df[cols].replace({'\$': '', ' ': '', ',': ''}, regex=True) # removes $ , and spaces
# stocks = df.sort_values(by = [8, 9], kind = "mergesort") # sorts by company --> by id
# stocks = stocks.to_numpy()
print(df.head())
df = df.apply(pd.to_numeric, errors='ignore', downcast='float')

stocks = df.sort_values(by = ["Company"], kind = "mergesort") # sorts by company --> by id
print(stocks.head())
#stocks.to_csv('my_array.csv')

minmax = MinMaxScaler().fit(df.iloc[:, 2:3].astype('float32')) # Close index
df_log = minmax.transform(df.iloc[:, 2:3].astype('float32')) # Close index
df_log = pd.DataFrame(df_log)
print(df_log.head())





'''cnt = 0
c = -1
visited = []
sortedStocks = [] 
for i in range(0, len(stocks[8])):
    if stocks[8][i] not in visited: # unique
        visited.append(stocks[8][i])
        cnt += 1
        c += 1
        sortedStocks[c].append(stocks[i])
    else:
        sortedStocks[c].append(stocks[i])
  
print("No.of.unique values :", cnt)
print("unique values :", visited)
print(sortedStocks)'''