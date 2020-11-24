import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

############################## Import data
train_wf_all = np.loadtxt('../../train_wf.csv',delimiter=',')
train_label_all = np.loadtxt('../../train_label.csv',delimiter=',')
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

############################## Recurrent Neural Network
model = Sequential([keras.Input((5000,)), 
					layers.LSTM(64),
					layers.LSTM(32),
					Dense(16, activation='relu'),
					Dense(2, activation='softmax')
					])

model.compile(optimizer = Adam(),
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

hist = model.fit(train_wf, train_label,
				 validation_data=(val_wf, val_label),
				 batch_size=1,
				 epochs=20,
				 shuffle=True,
				 verbose=2)

model.summary()
