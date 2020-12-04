import numpy as np
import h5py
import matplotlib.pyplot as plt

# Defining variables
print('Saving variable')
Path1 = '../../../Data/Electron40GeV/SmallFall/'
FileNameData1 = 'waveforms.hdf5'
Path2 = '../../../Data/Pion40GeV/SmallFall/'
FileNameData2 = 'waveforms.hdf5'

# Defining variables output
train_wf_0_ch = np.array([])
train_label_0_ch = np.array([])
train_wf_1_ch = np.array([])
train_label_1_ch = np.array([])
train_wf_2_ch = np.array([])
train_label_2_ch = np.array([])
train_wf_0_sc = np.array([])
train_label_0_sc = np.array([])
train_wf_1_sc = np.array([])
train_label_1_sc = np.array([])
train_wf_2_sc = np.array([])
train_label_2_sc = np.array([])

######################################################
########### --- Lettura file Elettroni --- ###########
######################################################

# File .h5py imput
print('Importing h5py file1')
with h5py.File(Path1 + FileNameData1, "r") as finwave:
	SiPMGeometry = np.array(finwave.get('Geometry'))
	DemoWaveform = np.array(finwave.get('Waveforms')[0])

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
	wavesEv_ch = np.array([])
	wavesEv_sc = np.array([])
	with h5py.File(Path1 + FileNameData1, "r") as finwave:
		wavesEv = np.array(finwave.get('Waveforms')[Minpos:Maxpos])
		Bool_ch = np.array([])
		Bool_ch = SiPMGeometry[:,1] == 0
		Bool_sc = np.array([])
		Bool_sc = SiPMGeometry[:,1] == 1
		wavesEv_ch = wavesEv[Bool_ch[Minpos:Maxpos]] #filtra su Cherenkov
		wavesEv_sc = wavesEv[Bool_sc[Minpos:Maxpos]] #filtra su Scintillanti
	sumAnalog_ch = wavesEv_ch.sum(axis=0)
	sumAnalog_sc = wavesEv_sc.sum(axis=0)
	del wavesEv
	del wavesEv_ch
	del wavesEv_sc
	label = 0
	train_wf_0_ch = np.append(train_wf_0_ch, sumAnalog_ch)
	train_label_0_ch = np.append(train_label_0_ch, label)
	train_wf_0_sc = np.append(train_wf_0_sc, sumAnalog_sc)
	train_label_0_sc = np.append(train_label_0_sc, label)

train_wf_0_ch = train_wf_0_ch.reshape(int(len(train_wf_0_ch)/DemoWaveform.size), DemoWaveform.size)
train_wf_0_sc = train_wf_0_sc.reshape(int(len(train_wf_0_sc)/DemoWaveform.size), DemoWaveform.size)

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

for ev in events:
	#gira nelle righe e somma le waveform per eventi
	print('File2: ' + str(ev) + " " + str(np.max(events)))
	BoolEv = np.array([])
	BoolEv = SiPMGeometry[:,0] == ev
	Pos = np.where(BoolEv)
	del BoolEv
	Minpos = np.min(Pos)
	Maxpos = np.max(Pos)
	wavesEv = np.array([])
	wavesEv_ch = np.array([])
	wavesEv_sc = np.array([])
	with h5py.File(Path2 + FileNameData2, "r") as finwave:
		wavesEv = np.array(finwave.get('Waveforms')[Minpos:Maxpos])
		Bool_ch = np.array([])
		Bool_ch = SiPMGeometry[:,1] == 0
		Bool_sc = np.array([])
		Bool_sc = SiPMGeometry[:,1] == 1
		wavesEv_ch = wavesEv[Bool_ch[Minpos:Maxpos]] #filtra su Cherenkov
		wavesEv_sc = wavesEv[Bool_sc[Minpos:Maxpos]] #filtra su Scintillanti
	sumAnalog_ch = wavesEv_ch.sum(axis=0)
	sumAnalog_sc = wavesEv_sc.sum(axis=0)
	del wavesEv
	del wavesEv_ch
	del wavesEv_sc
	label = 1
	train_wf_1_ch = np.append(train_wf_1_ch, sumAnalog_ch)
	train_label_1_ch = np.append(train_label_1_ch, label)
	train_wf_1_sc = np.append(train_wf_1_sc, sumAnalog_sc)
	train_label_1_sc = np.append(train_label_1_sc, label)

train_wf_1_ch = train_wf_1_ch.reshape(int(len(train_wf_1_ch)/DemoWaveform.size), DemoWaveform.size)
train_wf_1_sc = train_wf_1_sc.reshape(int(len(train_wf_1_sc)/DemoWaveform.size), DemoWaveform.size)

#################################################
### --- Creazione dati Elettrone + Pione --- ####
#################################################

#print(len(train_label_0_ch))
one_type_size = len(train_label_0_ch)

for ev in range(one_type_size):
	print('Sovrapposizione: ' + str(ev) + " " + str(one_type_size))
	label = 2
	superposition_ch = train_wf_0_ch[ev] + train_wf_1_ch[ev]
	superposition_sc = train_wf_0_sc[ev] + train_wf_1_sc[ev]
	print(len(superposition_ch))
	print(len(superposition_sc))
	train_wf_2_ch = np.append(train_wf_2_ch, superposition_ch)
	train_label_2_ch = np.append(train_label_2_ch, label)
	train_wf_2_sc = np.append(train_wf_2_sc, superposition_sc)
	train_label_2_sc = np.append(train_label_2_sc, label)

train_wf_2_ch = train_wf_2_ch.reshape(int(len(train_wf_2_ch)/DemoWaveform.size), DemoWaveform.size)
train_wf_2_sc = train_wf_2_sc.reshape(int(len(train_wf_2_sc)/DemoWaveform.size), DemoWaveform.size)

#################################################
####### --- Normalizazione e reshape --- ########
#################################################

train_wf_ch = train_wf_0_ch
train_label_ch = train_label_0_ch
train_wf_sc = train_wf_0_sc
train_label_sc = train_label_0_sc

train_wf_ch = np.append(train_wf_ch,train_wf_1_ch,axis=0)
train_wf_ch = np.append(train_wf_ch,train_wf_2_ch,axis=0)
train_label_ch = np.append(train_label_ch,train_label_1_ch,axis=0)
train_label_ch = np.append(train_label_ch,train_label_2_ch,axis=0)
train_wf_sc = np.append(train_wf_sc,train_wf_1_sc,axis=0)
train_wf_sc = np.append(train_wf_sc,train_wf_2_sc,axis=0)
train_label_sc = np.append(train_label_sc,train_label_1_sc,axis=0)
train_label_sc = np.append(train_label_sc,train_label_2_sc,axis=0)

print('Cher shape: ' + str(train_wf_ch.shape))
print('Cher label: ' + str(train_label_ch.shape))
print('Scin shape: ' + str(train_wf_sc.shape))
print('Scin label: ' + str(train_label_sc.shape))

normFactor_ch = np.max(train_wf_ch)
train_wf_ch = train_wf_ch / normFactor_ch
normFactor_sc = np.max(train_wf_sc)
train_wf_sc = train_wf_sc / normFactor_sc

##############################################
########### --- Scrittura .csv --- ###########
##############################################
''' SHUFFLER MEGLIO USARLO IN LETTURA
shuffler = np.random.permutation(len(train_wf))
train_wf = train_wf[shuffler]
train_label = train_label[shuffler]
'''
np.savetxt('../../train_wf_ch.csv',train_wf_ch,delimiter=',')
np.savetxt('../../train_label_ch.csv',train_label_ch,delimiter=',')

#np.savetxt('../../train_wf_sc.csv',train_wf_sc,delimiter=',')
#np.savetxt('../../train_label_sc.csv',train_label_sc,delimiter=',')

