import numpy as np
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy, sparse_categorical_crossentropy

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

import matplotlib.pyplot as plt

import pickle

'''
train_wf_all = np.loadtxt('train_wf.csv',delimiter=',')
train_label_all = np.loadtxt('train_label.csv',delimiter=',')
print(train_wf_all.shape)
print(train_label_all.shape)

shuffler = np.random.permutation(len(train_label_all))
train_wf_all = train_wf_all[shuffler]
train_label_all = train_label_all[shuffler]

val_wf = train_wf_all[1800:]
val_label = train_label_all[1800:]
train_wf = train_wf_all[:1800]
train_label = train_label_all[:1800]

print(train_wf.shape)
print(train_label.shape)
print(val_wf.shape)
print(val_label.shape)
'''
##################################
def model_builder(hp):
	model = keras.Sequential()

	model.add(Dense(units=hp.Int('input_units_0',16,64,8), input_shape=(5000,)))
	model.add(Activation('relu'))
	#model.add(Dense(units=hp.Int(f'input_units_1',8,64,8)))
	#model.add(Activation('relu'))

	for i in range(hp.Int('n_layers',1,4)):
		model.add(Dense(units=hp.Int(f'units_{i}',8,64,8)))
		model.add(Activation('relu'))
	

	model.add(Dense(units=2))
	model.add(Activation('softmax'))
    
	lr = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
	
	model.compile(optimizer = Adam(lr),
				  loss = 'sparse_categorical_crossentropy',
				  metrics = ['accuracy'])

	return model
#################################

with open('HTHist.pkl','rb') as f:
	tun = pickle.load(f)

print(tun.results_summary())
print(tun.get_best_models(num_models=1)[0].summary())

best_model = tun.get_best_models(num_models=1)[0]

hist = best_model.fit(x=train_wf, y=train_label, validation_split=0.05, epochs=400, shuffle=True, verbose=0)

best_model.summary()


acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epoch= range(1,401)

plt.plot(epoch, loss, label='loss')
plt.plot(epoch, val_loss, label='val_loss')
plt.legend()
plt.show()

plt.plot(epoch, acc, label='accuracy')
plt.plot(epoch, val_acc, label='val_accuracy')
plt.legend()
plt.show()
