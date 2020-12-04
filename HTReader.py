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
from sklearn.metrics import confusion_matrix
import seaborn as sns
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

##################################
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
#################################

############################## Best model and train
with open('HTHist_sc100DropAcc.pkl','rb') as f:
	tuned = pickle.load(f)

print(tuned.results_summary())
print(tuned.get_best_models(num_models=1)[0].summary())

best_model = tuned.get_best_models(num_models=1)[0]

hist = best_model.fit(x=train_wf, y=train_label, validation_split=0.1, epochs=400, shuffle=True, verbose=0, batch_size = 32)

best_model.summary()
valut = best_model.evaluate(val_wf,val_label,batch_size = 32)
print('loss,acc: ' + str(valut))

best_model.save('../../SavedModel/model_sc100DropAcc', overwrite=True)

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

############################## Prediction
prediction = best_model.predict(val_wf)
pred_label = np.argmax(prediction, axis=1)
minx = np.min(val_label) - 0.5
maxx = np.max(val_label) + 0.5
miny = np.min(pred_label) - 0.5
maxy = np.max(pred_label) + 0.5
#print(prediction[0])
#print(val_label[0])
#print(len(np.argmax(prediction, axis=1)))
#print(len(val_label))

############################## Plot
fig, ax = plt.subplots(figsize =(8, 6))

# create heatmap
conf_matrix = confusion_matrix(val_label, pred_label)
print(conf_matrix)
normalizer = conf_matrix.sum(axis=0)
print(normalizer)
conf_matrix_norm = conf_matrix / normalizer
print(conf_matrix_norm)
hmap = sns.heatmap(conf_matrix_norm, 
				   annot = True,
				   cmap = plt.cm.Blues,
				   fmt='.1%',
				   ax = ax)

# add labels
ax.set_xlabel('True label')
ax.set_ylabel('Predicted label')
ax.set_xticks([0.5,1.5,2.5])
ax.set_xticklabels(['$e^-$','$\pi^-$','$e^- + \pi^-$'])
plt.yticks([0.5,1.5,2.5], rotation=0, va='center')
ax.set_yticklabels(['$e^-$','$\pi^-$','$e^- + \pi^-$'])

# lavel colorbar
cbar = hmap.collections[0].colorbar
cbar.set_ticks([0.,0.25,0.5,0.75,1.])
cbar.set_ticklabels(['0.0%','25.0%','50.0%','75.0%','100.0%'])

# add border
for _, spine in hmap.spines.items():
    spine.set_visible(True)

plt.tight_layout()
plt.show()
