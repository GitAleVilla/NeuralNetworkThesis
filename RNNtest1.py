import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_crossentropy

############################## Import data
train_wf_all = np.loadtxt('../../train_wf_ch.csv',delimiter=',')
train_label_all = np.loadtxt('../../train_label_ch.csv',delimiter=',')
print(train_wf_all.shape)
print(train_label_all.shape)

shuffler = np.random.permutation(len(train_label_all))
train_wf_all = train_wf_all[shuffler]
train_label_all = train_label_all[shuffler]

data_size = len(train_label_all)
split_train_ratio = 0.9
split_number = int(data_size * split_train_ratio)

val_wf = train_wf_all[split_number:]
val_label = train_label_all[split_number:]
train_wf = train_wf_all[:split_number]
train_label = train_label_all[:split_number]

train_wf = np.reshape(train_wf, (train_wf.shape[0],train_wf.shape[1],1))
val_wf = np.reshape(val_wf, (val_wf.shape[0],val_wf.shape[1],1))

print(train_wf.shape)
print(val_wf.shape)

############################## Recurrent Neural Network
model = keras.Sequential([#keras.Input((5000,1,)), 
						  layers.LSTM(32, activation='relu', input_shape=(5000,1), return_sequences=False),
						  #layers.LSTM(32, activation='relu'),
						  #layers.Dense(16, activation='relu'),
						  layers.Dense(3, activation='softmax')
						  ])

model.compile(optimizer = Adam(1e-4),
			  #optimizer = 'sgd',
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

hist = model.fit(train_wf, train_label,
				 validation_data=(val_wf, val_label),
				 batch_size=128,
				 epochs=20,
				 shuffle=True,
				 verbose=1)

model.summary()
