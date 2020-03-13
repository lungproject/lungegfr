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

from Loaddata_allparttest import load_sampledata,load_sampledata2
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


modelpetct1 = load_model('./model/LungEGFR.hdf5')#,weightspatient2-improvement-40-0.67
modelpetct1.summary()

xpet_val,xct_val,xfuse_val = load_sampledata()


xpet_val= np.expand_dims(xpet_val, axis=3)
xct_val = np.expand_dims(xct_val, axis=3)
xfuse_val = np.expand_dims(xfuse_val, axis=3)

x_val = np.concatenate((xpet_val,xct_val,xfuse_val),axis=3)

predictvalpetct1 = modelpetct1.predict(x_val, verbose=1)

np.savetxt("./Results/Hebeipredicttest_3.txt",predictvalpetct1 )

xpet_val,xct_val,xfuse_val = load_sampledata2()


xpet_val= np.expand_dims(xpet_val, axis=3)
xct_val = np.expand_dims(xct_val, axis=3)
xfuse_val = np.expand_dims(xfuse_val, axis=3)

x_val = np.concatenate((xpet_val,xct_val,xfuse_val),axis=3)

predictvalpetct1 = modelpetct1.predict(x_val, verbose=1)

np.savetxt("./Results/Hebeipredicttest_5.txt",predictvalpetct1 )


# model = Model(inputs=modelpetct1.input,
#                 outputs=modelpetct1.get_layer('activation_8').output)#创建的新模型

# activations = model.predict([xpet_val,xct_val,xfuse_val])#W
# print(activations.shape)
# np.save('activations8.npy', activations)

# model = Model(inputs=modelpetct1.input,
#                 outputs=modelpetct1.get_layer('activation_18').output)#创建的新模型

# activations = model.predict([xpet_val,xct_val,xfuse_val])#W
# print(activations.shape)
# np.save('activations18.npy', activations)