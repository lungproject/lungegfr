import os  
from PIL import Image  
import csv
import numpy as np  
from keras import backend as K
import scipy.io
 


def load_SPHtraindata():
    
    img = np.load("./Alldata/sphxptrainpet2_64.npy") #Deeplearningallpatchsmalls
    datapet = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/sphxptrainct2_64.npy") #Deeplearningallpatchsmalls
    datact = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/sphxptrainfuse2_64.npy") #Deeplearningallpatchsmalls
    datafuse = np.asarray(img,dtype="float32")
    y = np.load("./Alldata/sphytrain.npy")
    label = np.asarray(y,dtype="float32")
    return datapet,datact,datafuse,label
    

def load_SPHtestdata():
    
    img = np.load("./Alldata/sphxptestpet2_64.npy") #Deeplearningallpatchsmalls
    datapet = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/sphxptestct2_64.npy") #Deeplearningallpatchsmalls
    datact = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/sphxptestfuse2_64.npy") #Deeplearningallpatchsmalls
    datafuse = np.asarray(img,dtype="float32")
    y = np.load("./Alldata/sphytest.npy")
    label = np.asarray(y,dtype="float32")
    return datapet,datact,datafuse,label


def load_Hebeitraindata():
    
    img = np.load("./Alldata/hebeixptrainpet2_64.npy") #Deeplearningallpatchsmalls
    datapet = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/hebeixptrainct2_64.npy") #Deeplearningallpatchsmalls
    datact = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/hebeixptrainfuse2_64.npy") #Deeplearningallpatchsmalls
    datafuse = np.asarray(img,dtype="float32")
    y = np.load("./Alldata/hebeiytrain.npy")
    label = np.asarray(y,dtype="float32")
    return datapet,datact,datafuse,label
    

def load_Hebeitestdata():
    
    img = np.load("./Alldata/hebeixptestpet2_64.npy") #Deeplearningallpatchsmalls
    datapet = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/hebeixptestct2_64.npy") #Deeplearningallpatchsmalls
    datact = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/hebeixptestfuse2_64.npy") #Deeplearningallpatchsmalls
    datafuse = np.asarray(img,dtype="float32")
    y = np.load("./Alldata/hebeiytest.npy")
    label = np.asarray(y,dtype="float32")
    return datapet,datact,datafuse,label    


def load_hlmtestdata():
    
    img = np.load("./Alldata/hlmxppet2s_64.npy") #Deeplearningallpatchsmalls
    datapet = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/hlmxpct2s_64.npy") #Deeplearningallpatchsmalls
    datact = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/hlmxpfuse2s_64.npy") #Deeplearningallpatchsmalls
    datafuse = np.asarray(img,dtype="float32")   
    return datapet,datact,datafuse



def load_Harbintestdata():
    
    img = np.load("./Alldata/harbinxptestpet2_64.npy") #Deeplearningallpatchsmalls
    datapet = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/harbinxptestct2_64.npy") #Deeplearningallpatchsmalls
    datact = np.asarray(img,dtype="float32")
    img = np.load("./Alldata/harbinxptestfuse2_64.npy") #Deeplearningallpatchsmalls
    datafuse = np.asarray(img,dtype="float32")
    y = np.load("./Alldata/harbinytest.npy")
    label = np.asarray(y,dtype="float32")

    return datapet,datact,datafuse,label




