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

stonk = [] # list of data frames 0-9 for 10 companies
company_list = [] # list of companies (10)
# company_list=stocks['Company'].unique().tolist()

for company, stockCompany in stocks.groupby('Company'):
    company_list.append(company)
    stonk.append(stockCompany)
    
trainlen = 100
future = 1
futureTotal = 100
pred_tot = np.zeros(futureTotal)
# maybe just focus on high?
compJ = []
print(stonk[0])
print(type(stonk[0]))
for i in stonk[0]:
    print(type(i))
    compJ.append(i['High'])

print(compJ)
for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(trainlen), stonk[0][i:trainlen+i])
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