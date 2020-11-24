import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

############################## Import model
model = keras.models.load_model('../../SavedModel/my_model')

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
x_bins = np.linspace(minx, maxx, 3) 
y_bins = np.linspace(miny, maxy, 3)
print(x_bins)
print(y_bins)

fig, ax = plt.subplots(figsize =(8, 6))

plt.hist2d(x = val_label, y = pred_label,
		   bins = [x_bins, y_bins],
		   cmap = plt.cm.Blues)

#print(xbins)

ax.set_xlabel('True label')
ax.set_ylabel('Predicted label')
ax.set_xticks([0,1,2])
ax.set_xticklabels(['$e^-$','$\pi^-$','$e^- + \pi^-$'])
ax.set_yticks([0,1,2])
ax.set_yticklabels(['$e^-$','$\pi^-$','$e^- + \pi^-$'])

plt.colorbar()
plt.tight_layout()
plt.show()

