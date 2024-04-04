from msilib import sequence
from pyexpat import model
import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split

bodyswing_df = pd.read_csv("BODYSWING.txt")
handswing_df = pd.read_csv("HANDSWING.txt")

no_of_timesteps = 10
x=[]
y=[]
dataset = bodyswing_df.iloc[:,1:].values  #lay tat ca cac dong bo cot 1
n_sample = len(dataset)

for i in range(no_of_timesteps, n_sample):
    x.append(dataset[i-no_of_timesteps:i,:])
    y.append(1)

dataset = handswing_df.iloc[:,1:].values  #lay tat ca cac dong bo cot 1
n_sample = len(dataset)

for i in range(no_of_timesteps, n_sample):
    x.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)

x,y = np.array(x), np.array(y)
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model  = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1, activation="sigmoid"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")

model.fit(x_train, y_train, epochs=16, batch_size=32,validation_data=(x_test, y_test))
model.save("model.h5")
