import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy, sparse_categorical_crossentropy

from sklearn.metrics import confusion_matrix
import seaborn as sns

def gaussian(x, mu, sig):
	return np.exp(-np.power(x - mu,2.) / (2 * np.power(sig,2.)))

##################################
def model_builder(hp):
	model = keras.Sequential()

	model.add(Dense(units=hp.Int('input_units_0',16,64,8), input_shape=(5000,)))
	model.add(Activation('relu'))

	for i in range(hp.Int('n_layers',1,8)):
		model.add(Dense(units=hp.Int(f'units_{i}',8,64,8)))
		model.add(Activation('relu'))

	model.add(Dense(units=3))
	model.add(Activation('softmax'))
    
	lr = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
	
	model.compile(optimizer = Adam(lr),
				  loss = 'sparse_categorical_crossentropy',
				  metrics = ['accuracy'])

	return model
################### GENERAZIONE DATI
train_wf = np.array([])
train_label = np.array([])
val_wf = np.array([])
val_label = np.array([])
x_val = np.linspace(0,500,5000)
SAMPLE_SIZE = 3000

train_label = np.random.randint(3,size = SAMPLE_SIZE)
val_label = np.random.randint(3,size = SAMPLE_SIZE)

for ev in range(SAMPLE_SIZE):
	mu = (np.random.rand() * 20) + 50
	sig = (np.random.rand() * 5) + 10
	train_wf = np.append(train_wf,gaussian(x_val,mu,sig))
	val_wf = np.append(val_wf,gaussian(x_val,mu,sig))

train_wf = train_wf.reshape(int(len(train_wf)/5000), 5000)
val_wf = val_wf.reshape(int(len(val_wf)/5000), 5000)

print(train_wf.shape)
print(train_label.shape)
#print(train_label)
print(val_wf.shape)
print(val_label.shape)
#print(val_label)

'''for i in range(len(train_label)):
	plt.plot(x_val,train_wf[i])
	plt.show()'''

################### RETE NEURALE
with open('HTHist_ch1000.pkl','rb') as f:
	tuned = pickle.load(f)

best_model = tuned.get_best_models(num_models=1)[0]
hist = best_model.fit(x=train_wf, y=train_label, validation_split=0.1, epochs=400, shuffle=True, verbose=2, batch_size = 64)

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

################### PREDICTION
prediction = best_model.predict(val_wf)
pred_label = np.argmax(prediction, axis=1)
minx = np.min(val_label) - 0.5
maxx = np.max(val_label) + 0.5
miny = np.min(pred_label) - 0.5
maxy = np.max(pred_label) + 0.5

################### VALIDATION
'''x_bins = np.linspace(minx, maxx, 4) 
y_bins = np.linspace(miny, maxy, 4)
print(x_bins)
print(y_bins)'''

fig, ax = plt.subplots()

# create heatmap
conf_matrix = confusion_matrix(pred_label, val_label)
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

'''plt.hist2d(x = val_label, y = pred_label,
		   bins = [x_bins, y_bins],
		   cmap = plt.cm.Blues)'''

# add labels
ax.set_xlabel('True label')
ax.set_ylabel('Predicted label')
ax.set_xticks([0.5,1.5,2.5])
ax.set_xticklabels(['$e^-$','$\pi^-$','$e^- + \pi^-$'])
plt.yticks([0.5,1.5,2.5], rotation=0, va='center')
ax.set_yticklabels(['$e^-$','$\pi^-$','$e^- + \pi^-$'])

#cbar = hmap.collections[0].colorbar
#cbar.set_ticks([0.,0.25,0.5,0.75,1.])
#cbar.set_ticklabels(['0.0%','25.0%','50.0%','75.0%','100.0%'])

# add border
for _, spine in hmap.spines.items():
    spine.set_visible(True)

#plt.colorbar()
plt.tight_layout()
plt.show()


#np.savetxt('../../dummy_wf.csv',train_wf,delimiter=',')
#np.savetxt('../../dummy_label.csv',train_label,delimiter=',')
