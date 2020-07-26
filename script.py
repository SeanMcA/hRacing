# Import the necessary packages
import pandas as pd
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler



file1 = 'racing/5.binaryEncodedOutput.csv'
file2 = "/input/5.binaryEncodedOutput.csv" # this is the data set uploaded to floydhub
df = pd.read_csv(file2,sep=',', engine='python',  header=0)

target = 'target'


x_train, x_test, y_train, y_test = train_test_split(df,df[target], test_size=0.2,random_state=2018) 


x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.1, random_state=2018)


scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)


model = Sequential()
model.add(Dense(512,input_dim = x_train.shape[1],activation="relu"))
model.add(Dense(512,activation="relu"))
model.add(Dense(1,activation = "sigmoid")) 
model.compile(optimizer = "Adam",loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train,y_train, validation_data = (x_val,y_val),epochs=5, batch_size=8)