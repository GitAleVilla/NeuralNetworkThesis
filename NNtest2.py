import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

import matplotlib.pyplot as plt

#####################################
#######---Lettura dati---############
#####################################

train_wf = np.loadtxt('train_wf.csv',delimiter=',')
train_label = np.loadtxt('train_label.csv',delimiter=',')

train_1 = train_wf[np.where(train_label == 0)]
train_2 = train_wf[np.where(train_label == 1)]
train_3 = np.array([])

train_lb1 = train_label[np.where(train_label == 0)]
train_lb2 = train_label[np.where(train_label == 1)] 
train_lb3 = np.array([])

train_wf_def = np.array([])
train_label_def = np.array([])

train_wf_def = np.append(train_wf_def,train_1[:800])
train_label_def = np.append(train_label_def,train_lb1[:800])
train_wf_def = np.append(train_wf_def,train_2[:800])
train_label_def = np.append(train_label_def,train_lb2[:800])

wf_superpos = np.array([])
for pos in range(800, len(train_1)):
	wf_superpos = train_1[pos] + train_2[pos]
	wf_superpos = wf_superpos / np.max(wf_superpos)
	train_3 = np.append(train_3,wf_superpos)
	label = 2
	train_lb3 = np.append(train_lb3,label)


train_wf_def = np.append(train_wf_def,train_3)
train_label_def = np.append(train_label_def,train_lb3)

train_wf_def = train_wf_def.reshape(len(train_label_def),5000)
#print(train_wf_def.shape)


shuffler = np.random.permutation(len(train_label_def))
train_wf_def = train_wf_def[shuffler]
train_label_def = train_label_def[shuffler]

#####################################
#######---Rete neurale---############
#####################################

model = Sequential([
    Dense(units=16,input_shape=(5000,),activation='relu'),
    Dense(units=16,activation='relu'),
    #Dense(units=16,activation='relu'),
    Dense(units=3,activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x=train_wf, y=train_label, validation_split=0.05, epochs=200, shuffle=True, verbose=2)

model.summary()

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epoch= range(1,201)

plt.plot(epoch, loss, label='loss')
plt.plot(epoch, val_loss, label='val_loss')
plt.legend()
plt.show()

