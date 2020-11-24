import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

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

############################## Model Constructor
def model_builder(hp):
	
	Rec = hp.Choice('Rec', values = [True,False])

	model = Sequential([])
	model.add(keras.Input((5000,)))
	if Rec:
		model.add(layers.LSTM(32))
	else:
		model.add(layers.Dense(32, activation('relu'))
	model.add(layers.Dense(16, activation('relu'))
	model.add(3, activation = 'softmax')
	model.compile(optimizer = Adam(),
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])
	return model

############################## HyperTuner
tuner = RandomSearch(model_builder,
					 objective = 'val_loss',
					 #objective = 'val_accuracy',
					 max_trials = 2,
					 executions_per_trial = 15,
					 directory = '../../HTRec_History')

tuner.search(x = train_wf,
             y = train_label,
             epochs = 200,
			 batch_size = 32,
             validation_data = (val_wf, val_label))


with open('HTHistRec.pkl','wb') as f:
	pickle.dump(tuner,f)
