# Axel Agelii, Axel Agelii
# 1, 1

import numpy as np
import pandas as pd
# This is the library for the Reservoir Computing got it by: https://github.com/cknd/pyESN
from pyESN import ESN
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

date_to_find = input("Enter Date(YYYY-MM-DD) to predict: ")
openVal = input("Enter open price (Just the number) for the previous day: ")
companyP = input("Enter the comapny name: ")


# obtaining training data
df = pd.read_csv('train.csv', sep=',')
# cleaning
cols = ["Close/Last", "Open", "High", "Low", "NextOpen"]
df[cols] = df[cols].replace({'\$': '', ' ': '', ',': ''}, regex=True) # removes $ , and spaces
df = df.apply(pd.to_numeric, errors='ignore', downcast='float')
stocks = df.sort_values(by = ["Company"], kind = "mergesort") 

# setting up echo state network (ESN)
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

# changing data to datetime and sorting by date
for i in stonk:
    i['Date'] = pd.to_datetime(i['Date'])
    i['Date'] = i['Date'].dt.strftime('%Y-%m-%d')
    i = i.iloc[:, 1:]
    i = i.sort_values(by = 'Date')
    i = i.set_index('Date')

company_to_predict_index = company_list.index(companyP)
company_to_predict = stonk[company_to_predict_index]
company_to_predict['Date'] = pd.to_datetime(company_to_predict['Date'])
company_to_predict['Date'] = company_to_predict['Date'].dt.strftime('%Y-%m-%d')
company_to_predict = company_to_predict.iloc[:, 1:]
company_to_predict = company_to_predict.sort_values(by = 'Date')
company_to_predict = company_to_predict.set_index('Date')

company_to_predict_training_set = company_to_predict['Open']
company_to_predict_training_set = pd.DataFrame(company_to_predict_training_set)


trainlen = company_to_predict.index.get_loc(date_to_find)-1# how many days to learn
future = 1 # 1 day in future
train = company_to_predict_training_set.to_numpy()
pred_training = esn.fit(np.ones(trainlen),train[:trainlen])

prediction = esn.predict(np.ones(future)) # this is the prediction

print("The predicted open price for the next day is: {0:.2f}".format(prediction[0][0]))