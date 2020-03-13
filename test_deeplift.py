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

from Loaddata_allparttest import load_sampledata,load_SPHtraindata,load_SPHtestdata,load_Hebeitraindata,load_Hebeitestdata,load_hlmtestdata,load_Harbintestdata
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import shap

modelpetct1 = load_model('./model/LungEGFR.hdf5')#,weightspatient2-improvement-40-0.67
modelpetct1.summary()

xpet_test,xct_test,xfuse_test,y_test = load_Harbintestdata()

xpet_test= np.expand_dims(xpet_test, axis=3)
xct_test = np.expand_dims(xct_test, axis=3)
xfuse_test = np.expand_dims(xfuse_test, axis=3)
x_test = np.concatenate((xpet_test,xct_test,xfuse_test),axis=3)

def map2layer(x, layer):
    feed_dict = dict(zip([modelpetct1.layers[0].input], [x.copy()]))
    return K.get_session().run(modelpetct1.layers[layer].input, feed_dict)
e = shap.GradientExplainer(
    (modelpetct1.layers[0].input, modelpetct1.layers[-1].output),
    map2layer(x_test, 0),
    local_smoothing=0 # std dev of smoothing noise
)
shap_values,indexes = e.shap_values(map2layer(x_test, 0), ranked_outputs=1)

shap.image_plot(shap_values, x_val)
shap.DeepExplainer

np.save('ttttt.npy', shap_values)







from IntegratedGradients import integrated_gradients
print(np.shape(xpet_val))
ig = integrated_gradients(model)
predict1 = model.predict([xpet_val,xct_val,xfuse_val ], verbose=1)
pred = np.argmax(predict1[1])
ex = ig.explain([xpet_val[1,:,:,:],xct_val[1,:,:,:],xfuse_val [1,:,:,:]], outc=pred)
print(np.shape(ex))
th = max(np.abs(np.min([np.min(ex[0]), np.min(ex[1])])), np.abs(np.max([np.max(ex[0]), np.max(ex[1])])))
plt.subplot(1, 3, 1)
plt.imshow(ex[0][:,:,0], cmap="seismic", vmin=-1*th, vmax=th)
plt.xticks([],[])
plt.yticks([],[])
plt.show()

plt.subplot(1, 3, 2)
plt.imshow(ex[1][:,:,0], cmap="seismic", vmin=-1*th, vmax=th)
plt.xticks([],[])
plt.yticks([],[])
plt.show()

plt.subplot(1, 3, 3)
plt.imshow(ex[2][:,:,0], cmap="seismic", vmin=-1*th, vmax=th)
plt.xticks([],[])
plt.yticks([],[])
plt.show()




Sampleindex=[3261,8377,11477,254]

negheatmaps= np.empty((4,8,8),dtype="float32")
posheatmaps= np.empty((4,8,8),dtype="float32")

for i in range(4):
    x=x_train[Sampleindex[i]:Sampleindex[i+1],:,:,:]
    ggoutput = modelct.output[:,0]
    layer = modelct.get_layer('activation_18')
    grads = K.gradients(ggoutput,layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([modelct.input], [pooled_grads, layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for j in range(256):
        conv_layer_output_value[:, :, j] *= pooled_grads_value[j]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    negheatmaps[i,:,:]=heatmap

    ggoutput = modelct.output[:,1]
    layer = modelct.get_layer('activation_18')
    grads = K.gradients(ggoutput,layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([modelct.input], [pooled_grads, layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for j in range(256):
        conv_layer_output_value[:, :, j] *= pooled_grads_value[j]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    posheatmaps[i,:,:]=heatmap

np.save('positivefilter.npy', posheatmaps)
np.save('negativefilter.npy', negheatmaps)


layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_name = 'conv2d_1'
for filter_index in range(4):
# filter_index = 2  # can be any integer from 0 to 511, as there are 512 filters in that layer
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.inputs)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.inputs], [loss, grads])  

    input_img_data = np.random.random((1, 64, 64, 3)) * 20 + 128.

    
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * 1
    img = input_img_data[0]   

    plt.figure(figsize=(25,25))
    plt.imshow(img)
    plt.grid(False)
    plt.show() 




layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_name = 'conv2d_106'
for filter_index in range(256)
# filter_index = 2  # can be any integer from 0 to 511, as there are 512 filters in that layer
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.inputs[0])[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.inputs[0]], [loss, grads])  

    input_img_data = np.random.random((1, 64, 64, 3)) * 20 + 128.                           
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * 1
    img = input_img_data[0]   

    plt.figure(figsize=(25,25))
    plt.imshow(img)
    plt.grid(False)


ind = 2433 #2834
ggoutput = model.output[:,0]
layer = model.get_layer('activation_76')
grads = K.gradients(ggoutput,layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x_val[ind:ind+1,:,:,:]])
for i in range(256):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)
negheatmap = np.maximum(heatmap, 0)
negheatmap /= np.max(negheatmap)
plt.matshow(heatmap)
plt.show()

ggoutput = model.output[:,1]
grads = K.gradients(ggoutput,layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x_val[ind:ind+1,:,:,:]])
for i in range(256):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)
posheatmap = np.maximum(heatmap, 0)
posheatmap /= np.max(posheatmap)





th= np.maximum(np.max(negheatmap),np.max(posheatmap))


plt.subplot(2, 2,2)
plt.imshow(x_val[ind:ind+1,:,:,0])
plt.xticks([],[])
plt.yticks([],[])
# plt.show()

plt.subplot(2, 2,1)
plt.imshow(negheatmap,vmin=0, vmax=th)
plt.show()

plt.subplot(2, 2,4)
plt.imshow(x_val[ind:ind+1,:,:,1])
plt.xticks([],[])
plt.yticks([],[])

plt.subplot(2, 2,3)
plt.imshow(posheatmap,vmin=0, vmax=th)
plt.show()     
# plt.matshow(img)
# plt.show()


# predicttestpetct1 = modelpetct1.predict(x_test, verbose=1)
# np.savetxt("./Results/Harbinpredicttest.txt",predicttestpetct1 )

# model = Model(inputs=modelpetct1.input,
#                 outputs=modelpetct1.get_layer('activation_76').output)#创建的新模型



# xpet_train,xct_train,xfuse_train,y_train = load_SPHtraindata()

# xpet_train= np.expand_dims(xpet_train, axis=3)
# xct_train = np.expand_dims(xct_train, axis=3)
# xfuse_train = np.expand_dims(xfuse_train, axis=3)

# x_train = np.concatenate((xpet_train,xct_train,xfuse_train),axis=3)
# predicttrainpetct1 = modelpetct1.predict(x_train, verbose=1)
# np.savetxt("./Results/SPHpredicttrain.txt",predicttrainpetct1 )

# activations = model.predict(x_train )#W
# print(activations.shape)
# np.save('./Results/activations76sphtrain.npy', activations)


# xpet_test,xct_test,xfuse_test,y_test = load_SPHtestdata()

# xpet_test= np.expand_dims(xpet_test, axis=3)
# xct_test = np.expand_dims(xct_test, axis=3)
# xfuse_test = np.expand_dims(xfuse_test, axis=3)
# x_test = np.concatenate((xpet_test,xct_test,xfuse_test),axis=3)

# predicttestpetct1 = modelpetct1.predict(x_test, verbose=1)
# np.savetxt("./Results/SPHpredicttest.txt",predicttestpetct1 )

# activations = model.predict(x_test )#W
# print(activations.shape)
# np.save('./Results/activations76sphtest.npy', activations)



# xpet_train,xct_train,xfuse_train,y_train = load_Hebeitraindata()

# xpet_train= np.expand_dims(xpet_train, axis=3)
# xct_train = np.expand_dims(xct_train, axis=3)
# xfuse_train = np.expand_dims(xfuse_train, axis=3)

# x_train = np.concatenate((xpet_train,xct_train,xfuse_train),axis=3)
# predicttrainpetct1 = modelpetct1.predict(x_train, verbose=1)
# np.savetxt("./Results/Hebeipredicttrain.txt",predicttrainpetct1 )

# activations = model.predict(x_train )#W
# print(activations.shape)
# np.save('./Results/activations76hebeitrain.npy', activations)



# xpet_test,xct_test,xfuse_test,y_test = load_Hebeitestdata()

# xpet_test= np.expand_dims(xpet_test, axis=3)
# xct_test = np.expand_dims(xct_test, axis=3)
# xfuse_test = np.expand_dims(xfuse_test, axis=3)
# x_test = np.concatenate((xpet_test,xct_test,xfuse_test),axis=3)

# predicttestpetct1 = modelpetct1.predict(x_test, verbose=1)
# np.savetxt("./Results/Hebeipredicttest.txt",predicttestpetct1 )

# activations = model.predict(x_test )#W
# print(activations.shape)
# np.save('./Results/activations76hebeitest.npy', activations)



# xpet_test,xct_test,xfuse_test,y_test = load_Harbintestdata()

# xpet_test= np.expand_dims(xpet_test, axis=3)
# xct_test = np.expand_dims(xct_test, axis=3)
# xfuse_test = np.expand_dims(xfuse_test, axis=3)
# x_test = np.concatenate((xpet_test,xct_test,xfuse_test),axis=3)

# predicttestpetct1 = modelpetct1.predict(x_test, verbose=1)
# np.savetxt("./Results/Harbinpredicttest.txt",predicttestpetct1 )


# activations = model.predict(x_test )#W
# print(activations.shape)
# np.save('./Results/activations76Harbintest.npy', activations)

xpet_test,xct_test,xfuse_test = load_hlmtestdata()

xpet_test= np.expand_dims(xpet_test, axis=3)
xct_test = np.expand_dims(xct_test, axis=3)
xfuse_test = np.expand_dims(xfuse_test, axis=3)
# x_test = np.concatenate((xpet_test,xct_test,xfuse_test),axis=3)
predicttestpetct1 = modelpetct1.predict([xpet_test,xct_test,xfuse_test], verbose=1)
np.savetxt("./Results/HLMPDLpredicttests.txt",predicttestpetct1 )


modelpetct1 = load_model('./model/LungEGFR.hdf5')#,weightspatient2-improvement-40-0.67
modelpetct1.summary()

model2 = Model(inputs=modelpetct1.input,
                outputs=modelpetct1.get_layer('activation_76').output)#创建的新模型

x_test = np.concatenate((xpet_test,xct_test,xfuse_test),axis=3)
predicttestpetct1 = modelpetct1.predict(x_test, verbose=1)
np.savetxt("./Results/HLMpredicttests.txt",predicttestpetct1 )
activations = model2.predict(x_test )#W
print(activations.shape)
np.save('./Results/activations76HLMtests.npy', activations)
