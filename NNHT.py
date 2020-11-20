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


train_wf_all = np.loadtxt('../../train_wf.csv',delimiter=',')
train_label_all = np.loadtxt('../../train_label.csv',delimiter=',')
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

#################################
def model_builder(hp):
	model = keras.Sequential()

	model.add(Dense(units=hp.Int('input_units_0',16,64,8), input_shape=(5000,)))
	model.add(Activation('relu'))

	for i in range(hp.Int('n_layers',1,8)):
		model.add(Dense(units=hp.Int(f'units_{i}',8,64,8)))
		model.add(Activation('relu'))
	

	model.add(Dense(units=2))
	model.add(Activation('softmax'))
    
	lr = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
	
	model.compile(optimizer = Adam(lr),
				  loss = 'sparse_categorical_crossentropy',
				  metrics = ['accuracy'])

	return model
##########################

tuner = RandomSearch(model_builder,
					 objective = 'val_loss',
					 #objective = 'val_accuracy',
					 max_trials = 25,
					 executions_per_trial = 3,
					 directory = '../../HT_History_batch4')

tuner.search(x = train_wf,
             y = train_label,
             epochs = 100,
			 batch_size = 4,
             validation_data = (val_wf, val_label))

print(train_wf.shape)
print(train_label.shape)
print(val_wf.shape)
print(val_label.shape)

with open('HTHist_batch4.pkl','wb') as f:
	pickle.dump(tuner,f)


