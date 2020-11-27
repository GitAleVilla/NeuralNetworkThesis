import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

############################## Import model
model = keras.models.load_model('../../SavedModel/model_cher1000')

############################## Import data
train_wf_all = np.loadtxt('../../train_wf_ch.csv',delimiter=',')
train_label_all = np.loadtxt('../../train_label_ch.csv',delimiter=',')
print(train_wf_all.shape)
print(train_label_all.shape)
print(np.unique(train_label_all))

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

############################## Prediction
prediction = model.predict(val_wf)
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
x_bins = np.linspace(minx, maxx, 4) 
y_bins = np.linspace(miny, maxy, 4)
print(x_bins)
print(y_bins)

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

cbar = hmap.collections[0].colorbar
cbar.set_ticks([0.,0.25,0.5,0.75,1.])
cbar.set_ticklabels(['0.0%','25.0%','50.0%','75.0%','100.0%'])

# add border
for _, spine in hmap.spines.items():
    spine.set_visible(True)

#plt.colorbar()
plt.tight_layout()
plt.show()

