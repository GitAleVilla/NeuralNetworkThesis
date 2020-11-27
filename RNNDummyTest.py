import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_crossentropy

############################## Import data
train_wf = np.array([])
train_label = np.array([])
val_wf = np.array([])
val_label = np.array([])

SAMPLE_TRAIN_SIZE = 500
SAMPLE_VAL_SIZE = 100
SAMPLE_LENGHT = 50

for ev in range(SAMPLE_TRAIN_SIZE):
	label = int(ev % 2)
	train_label = np.append(train_label,label)
	if label == 0:
		wf = np.zeros(SAMPLE_LENGHT,dtype=float)
	elif label == 1:
		wf = np.zeros(SAMPLE_LENGHT,dtype=float)
		wf[0] = 1.
	else:
		print('wtf?! Label not 0 nor 1')
	train_wf = np.append(train_wf,wf)

for ev in range(SAMPLE_VAL_SIZE):
	label = int(ev % 2)
	val_label = np.append(val_label,label)
	if label == 0:
		wf = np.zeros(SAMPLE_LENGHT,dtype=float)
	elif label == 1:
		wf = np.zeros(SAMPLE_LENGHT,dtype=float)
		wf[0] = 1.
	else:
		print('wtf?! Label not 0 nor 1')
	val_wf = np.append(val_wf,wf)

train_wf = np.reshape(train_wf, (SAMPLE_TRAIN_SIZE,SAMPLE_LENGHT,1))
val_wf = np.reshape(val_wf, (SAMPLE_VAL_SIZE,SAMPLE_LENGHT,1))

print(train_wf.shape)
print(val_wf.shape)

shuffler = np.random.permutation(len(val_label))
val_wf = val_wf[shuffler]
val_label = val_label[shuffler]

shuffler = np.random.permutation(len(train_label))
train_wf = train_wf[shuffler]
train_label = train_label[shuffler]

############################## Recurrent Neural Network
model = keras.Sequential([#keras.Input((5000,1,)), 
						  layers.LSTM(32, activation='relu', input_shape=(SAMPLE_LENGHT,1), return_sequences=False),
						  #layers.LSTM(32, activation='relu'),
						  #layers.Dense(16, activation='relu'),
						  layers.Dense(2, activation='softmax')
						  ])

model.compile(optimizer = Adam(1e-4),
			  #optimizer = 'sgd',
			  loss='sparse_categorical_crossentropy',
			  metrics=['accuracy'])

hist = model.fit(train_wf, train_label,
				 validation_data=(val_wf, val_label),
				 batch_size=128,
				 epochs=200,
				 shuffle=True,
				 verbose=1)

model.summary()

################ Plot loss and accuracy
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epoch= range(1,200+1)

plt.plot(epoch, loss, label='loss')
plt.plot(epoch, val_loss, label='val_loss')
plt.legend()
plt.show()

plt.plot(epoch, acc, label='accuracy')
plt.plot(epoch, val_acc, label='val_accuracy')
plt.legend()
plt.show()
