from __future__ import print_function

import datetime
import keras
import numpy as np

import os
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.models import  Sequential
from keras.layers import Input,Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D,BatchNormalization
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping,ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.constraints import maxnorm
from sklearn.cross_validation import StratifiedKFold

from keras import backend as K

from Loaddata_allparttest import load_SPHtraindata,load_SPHtestdata,load_Hebeitraindata,load_Hebeitestdata,load_hlmtestdata,load_Harbintestdata
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


modelpetct1 = load_model('./model/LungEGFR.hdf5')#,weightspatient2-improvement-40-0.67
modelpetct1.summary()




xpet_train,xct_train,xfuse_train,y_train = load_SPHtraindata()

xpet_train= np.expand_dims(xpet_train, axis=3)
xct_train = np.expand_dims(xct_train, axis=3)
xfuse_train = np.expand_dims(xfuse_train, axis=3)

x_train = np.concatenate((xpet_train,xct_train,xfuse_train),axis=3)
predicttrainpetct1 = modelpetct1.predict(x_train, verbose=1)
np.savetxt("./Results/SPHpredicttrain.txt",predicttrainpetct1 )

# activations = model.predict(x_train )#W
# print(activations.shape)
# np.save('./Results/activations76sphtrain.npy', activations)


xpet_test,xct_test,xfuse_test,y_test = load_SPHtestdata()

xpet_test= np.expand_dims(xpet_test, axis=3)
xct_test = np.expand_dims(xct_test, axis=3)
xfuse_test = np.expand_dims(xfuse_test, axis=3)
x_test = np.concatenate((xpet_test,xct_test,xfuse_test),axis=3)

predicttestpetct1 = modelpetct1.predict(x_test, verbose=1)
np.savetxt("./Results/SPHpredicttest.txt",predicttestpetct1 )

# activations = model.predict(x_test )#W
# print(activations.shape)
# np.save('./Results/activations76sphtest.npy', activations)



xpet_train,xct_train,xfuse_train,y_train = load_Hebeitraindata()

xpet_train= np.expand_dims(xpet_train, axis=3)
xct_train = np.expand_dims(xct_train, axis=3)
xfuse_train = np.expand_dims(xfuse_train, axis=3)

x_train = np.concatenate((xpet_train,xct_train,xfuse_train),axis=3)
predicttrainpetct1 = modelpetct1.predict(x_train, verbose=1)
np.savetxt("./Results/Hebeipredicttrain.txt",predicttrainpetct1 )

# activations = model.predict(x_train )#W
# print(activations.shape)
# np.save('./Results/activations76hebeitrain.npy', activations)



xpet_test,xct_test,xfuse_test,y_test = load_Hebeitestdata()

xpet_test= np.expand_dims(xpet_test, axis=3)
xct_test = np.expand_dims(xct_test, axis=3)
xfuse_test = np.expand_dims(xfuse_test, axis=3)
x_test = np.concatenate((xpet_test,xct_test,xfuse_test),axis=3)

predicttestpetct1 = modelpetct1.predict(x_test, verbose=1)
np.savetxt("./Results/Hebeipredicttest.txt",predicttestpetct1 )

# activations = model.predict(x_test )#W
# print(activations.shape)
# np.save('./Results/activations76hebeitest.npy', activations)



xpet_test,xct_test,xfuse_test,y_test = load_Harbintestdata()

xpet_test= np.expand_dims(xpet_test, axis=3)
xct_test = np.expand_dims(xct_test, axis=3)
xfuse_test = np.expand_dims(xfuse_test, axis=3)
x_test = np.concatenate((xpet_test,xct_test,xfuse_test),axis=3)

# predicttestpetct1 = modelpetct1.predict(x_test, verbose=1)
# np.savetxt("./Results/Harbinpredicttest.txt",predicttestpetct1 )

predicttestpetct1 = modelpetct1.predict([xpet_test,xct_test,xfuse_test], verbose=1)
np.savetxt("./Results/Harbinpdlpredicttest.txt",predicttestpetct1 )

# activations = model.predict(x_test )#W
# print(activations.shape)
# np.save('./Results/activations76Harbintest.npy', activations)

xpet_test,xct_test,xfuse_test = load_hlmtestdata()

xpet_test= np.expand_dims(xpet_test, axis=3)
xct_test = np.expand_dims(xct_test, axis=3)
xfuse_test = np.expand_dims(xfuse_test, axis=3)
x_test = np.concatenate((xpet_test,xct_test,xfuse_test),axis=3)
predicttestpetct1 = modelpetct1.predict([xpet_test,xct_test,xfuse_test], verbose=1)
np.savetxt("./Results/HLMpredicttests.txt",predicttestpetct1 )


# modelpetct1 = load_model('./model/LungEGFR.hdf5')#,weightspatient2-improvement-40-0.67
# modelpetct1.summary()

# model2 = Model(inputs=modelpetct1.input,
#                 outputs=modelpetct1.get_layer('activation_76').output)#创建的新模型

# x_test = np.concatenate((xpet_test,xct_test,xfuse_test),axis=3)
# predicttestpetct1 = modelpetct1.predict(x_test, verbose=1)
# np.savetxt("./Results/HLMpredicttests.txt",predicttestpetct1 )
# activations = model2.predict(x_test )#W
# print(activations.shape)
# np.save('./Results/activations76HLMtests.npy', activations)
