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


df = pd.read_csv('train.csv', sep=',')

cols = ["Close/Last", "Open", "High", "Low", "NextOpen"]
df[cols] = df[cols].replace({'\$': '', ' ': '', ',': ''}, regex=True) # removes $ , and spaces
df = df.apply(pd.to_numeric, errors='ignore', downcast='float')

stocks = df.sort_values(by = ["Company"], kind = "mergesort") 

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


for i in stonk:
    i['Date'] = pd.to_datetime(i['Date'])
    i['Date'] = i['Date'].dt.strftime('%Y-%m-%d')
    i = i.iloc[:, 1:]
    i = i.sort_values(by = 'Date')
    i = i.set_index('Date')

# trainlen = 1000 # change this; This is how many days in the past are we looking at
# future = 1
# futureTotal = 100 # change this??
# pred_tot = np.zeros(futureTotal)
trainlen = len(companyH_training_set) # length of what ever company we're looking at
future = 1
train = companyH_training_set.to_numpy() # the 'Open' for whatever company we're predicting for 
pred_training = esn.fit(np.ones(trainlen),train[:trainlen])
prediction = esn.predict(np.ones(future)) # this is the prediction for the next day
