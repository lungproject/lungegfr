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


from keras import backend as K

from Loaddata_allparttest import load_sampledata
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


modelpetct1 = load_model('./model/LungEGFR.hdf5')#,weightspatient2-improvement-40-0.67
modelpetct1.summary()



datapet,datact,datafuse,label = load_sampledata()


datapet = np.expand_dims(datapet, axis=3)
datact = np.expand_dims(datact, axis=3)
datafuse = np.expand_dims(datafuse, axis=3)

x_train = np.concatenate((datapet,datact,datafuse),axis=3)
predictpetct1 = modelpetct1.predict(x_train, verbose=1)
np.savetxt("./Results/Samplepredict.txt",predictpetct1)
