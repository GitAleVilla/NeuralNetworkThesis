import numpy as np
import h5py
import matplotlib.pyplot as plt

# Defining variables
print('Saving variable')
Path1 = '../../../Data/Electron40GeV/SmallFall/'
FileNameData1 = 'waveforms.hdf5'
Path2 = '../../../Data/Pion40GeV/SmallFall/'
FileNameData2 = 'waveforms.hdf5'


######################################################
########### --- Lettura file Elettroni --- ###########
######################################################

# File .h5py imput
print('Importing h5py file1')
with h5py.File(Path1 + FileNameData1, "r") as finwave:
	SiPMGeometry = np.array(finwave.get('Geometry'))
	DemoWaveform = np.array(finwave.get('Waveforms')[0])

# Defining variables Neural Network
train_wf_0 = np.array([])
train_label_0 = np.array([])
train_wf_1 = np.array([])
train_label_1 = np.array([])
train_wf_2 = np.array([])
train_label_2 = np.array([])

# np.array of events ID
events = SiPMGeometry[:,0]
events = np.unique(events)

for ev in events:
	#gira nelle righe e somma le waveform per eventi
	print('File1: ' + str(ev) + " " + str(np.max(events)))
	BoolEv = np.array([])
	BoolEv = SiPMGeometry[:,0] == ev
	Pos = np.where(BoolEv)
	del BoolEv
	Minpos = np.min(Pos)
	Maxpos = np.max(Pos)
	wavesEv = np.array([])
	with h5py.File(Path1 + FileNameData1, "r") as finwave:
		wavesEv = np.array(finwave.get('Waveforms')[Minpos:Maxpos])
	sumAnalog = wavesEv.sum(axis=0)
	del wavesEv
	label = 0
	train_wf_0 = np.append(train_wf_0, sumAnalog)
	train_label_0 = np.append(train_label_0, label)
	
print(train_wf_0.shape)
print(train_label_0.shape)

train_wf_0 = train_wf_0.reshape(int(len(train_wf_0)/DemoWaveform.size), DemoWaveform.size)

#################################################
########### --- Lettura file Pioni --- ##########
#################################################

# File .h5py imput
print('Importing h5py file2')
with h5py.File(Path2 + FileNameData2, "r") as finwave:
	SiPMGeometry = np.array(finwave.get('Geometry'))
	DemoWaveform = np.array(finwave.get('Waveforms')[0])

# np.array of events ID
events = SiPMGeometry[:,0]
events = np.unique(events)

for ev in events[:]:
	#gira nelle righe e somma le waveform per eventi
	print('File2: ' + str(ev) + " " + str(np.max(events)))
	BoolEv = np.array([])
	BoolEv = SiPMGeometry[:,0] == ev
	Pos = np.where(BoolEv)
	del BoolEv
	Minpos = np.min(Pos)
	Maxpos = np.max(Pos)
	wavesEv = np.array([])
	with h5py.File(Path2 + FileNameData2, "r") as finwave:
		wavesEv = np.array(finwave.get('Waveforms')[Minpos:Maxpos])
	sumAnalog = wavesEv.sum(axis=0)
	del wavesEv
	label = 1
	train_wf_1 = np.append(train_wf_1, sumAnalog)
	train_label_1 = np.append(train_label_1, label)

train_wf_1 = train_wf_1.reshape(int(len(train_wf_1)/DemoWaveform.size), DemoWaveform.size)

#################################################
### --- Creazione dati Elettrone + Pione --- ####
#################################################

one_type_size = len(train_label_0)

for ev in range(one_type_size):
	print('Sovrapposizione: ' + str(ev) + " " + str(one_type_size))
	label = 2
	superposition = train_wf_0[ev] + train_wf_1[ev]
	print(len(superposition))
	train_wf_2 = np.append(train_wf_2, superposition)
	train_label_2 = np.append(train_label_2, label)

train_wf_2 = train_wf_2.reshape(int(len(train_wf_2)/DemoWaveform.size), DemoWaveform.size)

#################################################
####### --- Normalizazione e reshape --- ########
#################################################

train_wf = train_wf_0
train_label = train_label_0

train_wf = np.append(train_wf,train_wf_1,axis=0)
train_wf = np.append(train_wf,train_wf_2,axis=0)
train_label = np.append(train_label,train_label_1,axis=0)
train_label = np.append(train_label,train_label_2,axis=0)


normFactor = np.max(train_wf)
train_wf = train_wf / normFactor

#print('len wf: ' + str(len(train_wf)))
#print('len label: ' + str(len(train_label)))
#print('shape wf: ' + str(train_wf.shape))
#print('shape label: ' + str(train_label.shape))

#train_wf = train_wf.reshape(int(len(train_wf)/DemoWaveform.size), DemoWaveform.size)
print(train_wf.shape)
print(train_label.shape)

##############################################
########### --- Scrittura .csv --- ###########
##############################################
'''
shuffler = np.random.permutation(len(train_wf))
train_wf = train_wf[shuffler]
train_label = train_label[shuffler]
'''
np.savetxt('../../train_wf.csv',train_wf,delimiter=',')
np.savetxt('../../train_label.csv',train_label,delimiter=',')

