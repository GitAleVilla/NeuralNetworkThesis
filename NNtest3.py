import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_crossentropy

import matplotlib.pyplot as plt

#####################################
#######---Lettura dati---############
#####################################

train_wf_all = np.loadtxt('../../train_wf.csv',delimiter=',')
train_label_all = np.loadtxt('../../train_label.csv',delimiter=',')
#train_wf_all = np.transpose(train_wf_all)
#train_label_all = np.transpose(train_label_all)
print(train_wf_all.shape)
print(train_label_all.shape)
'''
shuffler = np.random.permutation(len(train_label_all))
train_wf_all = train_wf_all[shuffler]
train_label_all = train_label_all[shuffler]

val_wf = train_wf_all[1800:]
val_label = train_label_all[1800:]
train_wf = train_wf_all[:1800]
train_label = train_label_all[:1800]

print(shuffler)
print(len(shuffler))

print(train_wf.shape)
print(train_label.shape)
print(val_wf.shape)
print(val_label.shape)
'''
#####################################
#######---Rete neurale---############
#####################################

model = Sequential([#keras.Input((5000,)), 
					Dense(16, activation='relu'),
					Dense(2, activation='softmax')
					])


model.compile(optimizer = Adam(),
			loss='sparse_categorical_crossentropy',
			#loss='mse',
			metrics=['accuracy'])

hist = model.fit(train_wf_all, train_label_all,
               #validation_data=(val_wf, val_label),
			   #validation_split=0.05,
			   batch_size=1,
			   epochs=20,
			   shuffle=True,
			   verbose=2)

model.summary()

print(train_wf.shape)
print(train_label.shape)
print(val_wf.shape)
print(val_label.shape)
