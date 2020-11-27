import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import sparse_categorical_crossentropy

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

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

print(train_wf.shape)
print(train_label.shape)
print(val_wf.shape)
print(val_label.shape)

############################## Model Constructor
def model_builder(hp):
	
	Rec = hp.Choice('Rec', values = [True,False])

	model = keras.Sequential()
	#model.add(keras.Input((5000,)))
	if Rec:
		model.add(layers.LSTM(32,input_shape=(5000,)))
		model.add(layers.Activation('relu'))
	else:
		model.add(layers.Dense(32,input_shape=(5000,)))
		model.add(layers.Activation('relu'))
	model.add(layers.Dense(16))
	model.add(layers.Activation('relu'))
	model.add(layers.Dense(3))
	model.add(layers.Activation('softmax'))
	model.compile(optimizer = Adam(),
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])
	return model

############################## HyperTuner
tuner = RandomSearch(model_builder,
					 objective = 'val_loss',
					 #objective = 'val_accuracy',
					 max_trials = 2,
					 executions_per_trial = 30,
					 directory = '../../HTRec_History',
					 project_name='RecurrentCherenkov')

tuner.search(x = train_wf,
             y = train_label,
             epochs = 200,
			 batch_size = 32,
             validation_data = (val_wf, val_label))


with open('HTHistRec.pkl','wb') as f:
	pickle.dump(tuner,f)
