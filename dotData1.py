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

# https://towardsdatascience.com/predicting-stock-prices-with-echo-state-networks-f910809d23d4
n_reservoir = 500
sparsity = 0.2
rand_seed = 23
spectral_radius = 1.2
noise = .0005


esn = ESN(n_inputs = 1,
      n_outputs = 1, 
      n_reservoir = n_reservoir,
      sparsity = sparsity,
      random_state = rand_seed,
      spectral_radius = spectral_radius,
      noise = noise)

visited = []
visited.append(stocks['Company'][0])
stonk = []
visited.append(stocks.iloc[0])
for i in range(1, len(stocks['Company'])):
    if stocks['Company'][i] not in visited: # unique
        break
    else:
        print(stocks.iloc[i])
        stonk.append(stocks.iloc[i])

# visited = visited.to_numpy()
print(stonk)
trainlen = 1500
future = 1
futureTotal = 100
pred_tot = np.zeros(futureTotal)

for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(trainlen), stocks[i:trainlen+i])
    prediction = esn.predict(np.ones(future))
    pred_tot[i:i+future] = prediction[:,0]

"""
error, validation_set = run_echo(1.2, .005,2)

future = 2
plt.figure(figsize=(18,8))
plt.plot(range(0,trainlen+future),stocks[0:trainlen+future],'k',label="target system")
plt.plot(range(trainlen,trainlen+100),validation_set.reshape(-1,1),'r', label="free running ESN")
lo,hi = plt.ylim()
plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
plt.legend(loc=(0.61,0.12),fontsize='x-large')
sns.despine()
"""