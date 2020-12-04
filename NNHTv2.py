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

####### - Import Data - ##########
train_wf_all = np.loadtxt('../../train_wf_sc.csv',delimiter=',')
train_label_all = np.loadtxt('../../train_label_sc.csv',delimiter=',')
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

print(train_wf.shape)
print(train_label.shape)
print(val_wf.shape)
print(val_label.shape)

######### - Model Constructor - ###########
def model_builder(hp):
	model = keras.Sequential()

	model.add(Dense(units=hp.Int('input_units_0',16,64,8), input_shape=(5000,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	for i in range(hp.Int('n_layers',1,3)):
		model.add(Dense(units=hp.Int(f'units_{i}',8,64,8)))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))

	model.add(Dense(units=3))
	model.add(Activation('softmax'))
    
	lr = hp.Choice('learning_rate', values = [1e-3, 1e-4, 1e-5])
	
	model.compile(optimizer = Adam(lr),
				  loss = 'sparse_categorical_crossentropy',
				  metrics = ['accuracy'])

	return model

############ - HyperTuner - ##############
tuner = RandomSearch(model_builder,
					 #objective = 'val_loss',
					 objective = 'val_accuracy',
					 max_trials = 100,
					 executions_per_trial = 3,
					 directory = '../../HT_History',
					 project_name='Scintillating100DropAcc')

tuner.search(x = train_wf,
             y = train_label,
             epochs = 100,
			 batch_size = 32,
             validation_data = (val_wf, val_label))

print(train_wf.shape)
print(train_label.shape)
print(val_wf.shape)
print(val_label.shape)

with open('HTHist_sc100DropAcc.pkl','wb') as f:
	pickle.dump(tuner,f)


