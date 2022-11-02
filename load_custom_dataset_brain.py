from __future__ import print_function
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras import backend as K
import gzip
import _pickle as cPickle
import sys
import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten
from tensorflow.keras.models import Sequential, Model
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from auto_cnn.gan import AutoCNN
from auto_cnn.cnn_structure import SkipLayer
from numpy import load
import random

tf.get_logger().setLevel('INFO')



import random

random.seed(42)
tf.random.set_seed(42)


import os
import cv2
import numpy as np
from numpy import load
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.get_logger().setLevel('INFO')



import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
random.seed(42)
tf.random.set_seed(42)


def mnist_test():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    values = x_train.shape[0] // 2

    data = {'x_train': x_train[:values], 'y_train': y_train[:values], 'x_test': x_test, 'y_test': y_test}

    a = AutoCNN(5, 1, data)
    a.run()

def adrenalmnist3d():

  data = load("data/adrenalmnist3d.npz")
  lst = data.files

  x_train,y_train,x_val,y_val,x_test,y_test = data['train_images'],data['train_labels'],data['val_images'],data['val_labels'],data['test_images'],data['test_labels']

  data = {'x_train': x_train, 'y_train': y_train,'x_val':x_val,'y_val':y_val, 'x_test': x_test, 'y_test': y_test}

  a= AutoCNN(18,7,data,'adrenalmnist3d','binary-class',epoch_number=10)
  a.run()

def bloodmnist():

  data = load("data/bloodmnist.npz")
  lst = data.files

  x_train,y_train,x_val,y_val,x_test,y_test = data['train_images'],data['train_labels'],data['val_images'],data['val_labels'],data['test_images'],data['test_labels']

  data = {'x_train': x_train, 'y_train': y_train,'x_val':x_val,'y_val':y_val, 'x_test': x_test, 'y_test': y_test}

  a = AutoCNN(5, 1, data,epoch_number=10)
  a.run()
  
def breastmnist():

  data = load("data/breastmnist.npz")
  lst = data.files

  x_train,y_train,x_val,y_val,x_test,y_test = data['train_images'],data['train_labels'],data['val_images'],data['val_labels'],data['test_images'],data['test_labels']

  data = {'x_train': x_train, 'y_train': y_train,'x_val':x_val,'y_val':y_val, 'x_test': x_test, 'y_test': y_test}

  a = AutoCNN(5, 1, data,epoch_number=10)
  a.run()
  
def fracturemnist3d():

  data = load("data/fracturemnist3d.npz")
  lst = data.files

  x_train,y_train,x_val,y_val,x_test,y_test = data['train_images'],data['train_labels'],data['val_images'],data['val_labels'],data['test_images'],data['test_labels']

  data = {'x_train': x_train, 'y_train': y_train,'x_val':x_val,'y_val':y_val, 'x_test': x_test, 'y_test': y_test}

  a = AutoCNN(5, 1, data)
  a.run()
  
def organcmnist():

  data = load("data/organcmnist.npz")
  lst = data.files

  x_train,y_train,x_val,y_val,x_test,y_test = data['train_images'],data['train_labels'],data['val_images'],data['val_labels'],data['test_images'],data['test_labels']

  data = {'x_train': x_train, 'y_train': y_train,'x_val':x_val,'y_val':y_val, 'x_test': x_test, 'y_test': y_test}
  a = AutoCNN(5, 1, data)
  a.run()

def pneumoniamnist():

  data = load("data/pneumoniamnist.npz")
  lst = data.files


  x_train,y_train,x_val,y_val,x_test,y_test = data['train_images'],data['train_labels'],data['val_images'],data['val_labels'],data['test_images'],data['test_labels']

  data = {'x_train': x_train, 'y_train': y_train,'x_val':x_val,'y_val':y_val, 'x_test': x_test, 'y_test': y_test}
  a = AutoCNN(5, 1, data)
  a.run()
  
def tissuemnist(file_path,task_name,epochs,task_type,pop_size,max_pop_size):

  data = load(file_path)
  x_train,y_train,x_val,y_val,x_test,y_test = data['train_images'],data['train_labels'],data['val_images'],data['val_labels'],data['test_images'],data['test_labels']

  data = {'x_train': x_train, 'y_train': y_train,'x_val':x_val,'y_val':y_val, 'x_test': x_test, 'y_test': y_test}
  a = AutoCNN(pop_size, max_pop_size,task_type,data,task_name, epoch_number=1)
  a.run()
  
  



def cifar10_test():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    values = x_train.shape[0]
    x_train,y_train,x_val,y_val = data['train_images'],data['train_labels'],data['val_images'],data['val_labels']
    data = {'x_train': x_train, 'y_train': y_train,'x_val':x_val,'y_val':y_val, 'x_test': x_test, 'y_test': y_test}
    a = AutoCNN(5, 1, data)
    a.run()

def read_dataset(file_path,task_name,epochs,task_type,pop_size,max_pop_size):

  data = load(file_path)
  x_train,y_train,x_val,y_val,x_test,y_test = data['train_images'],data['train_labels'],data['val_images'],data['val_labels'],data['test_images'],data['test_labels']

  data = {'x_train': x_train, 'y_train': y_train,'x_val':x_val,'y_val':y_val, 'x_test': x_test, 'y_test': y_test}
  a = AutoCNN(pop_size, max_pop_size,task_type,data,task_name, epoch_number=10)
  a.run()
  
IMG_WIDTH=128
IMG_HEIGHT=128

def create_dataset(img_folder):
   
    img_data_array=[]
    class_name=[]
   
    for dir1 in tqdm(os.listdir(img_folder)):
        for file in tqdm(os.listdir(os.path.join(img_folder, dir1))):
       
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_RGB2GRAY)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name

tf.get_logger().setLevel('INFO')





def read_custom_dataset(file_path,task_name,task_type,pop_size,max_pop_size,epochs):
    # extract the image array and class name
    img_data, class_name =create_dataset(file_path)
    target_dict={k: v for v, k in enumerate(np.unique(class_name))}
    target_dict
    target_val=  [target_dict[class_name[i]] for i in range(len(class_name))]
    img_data=np.asarray(img_data)
    target_val=np.asarray(target_val)

    x_train, x_test, y_train, y_test = train_test_split(img_data, target_val, test_size=0.33, random_state=42)
    print(y_test)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.33, random_state=42, stratify = y_test)
    values = x_train.shape[0] // 4

    data = {'x_train': x_train[:values], 'y_train': y_train[:values], 'x_test': x_test, 'y_test': y_test,'x_val': x_val, 'y_val': y_val}

    
    a = AutoCNN(pop_size, max_pop_size,task_type,data,task_name, epoch_number=epochs)
    a.run()    

if __name__ == '__main__':
  #read_dataset("data/pathmnist.npz","pathmnist","multi-label binary-class",18,19,10)
  #read_custom_dataset("archive (4).zip (Unzipped Files)/chest_xray/train/","chestdata","binary-class",18,19,10)
  read_custom_dataset("brain/","brain","binary-class",18,19,10)
  #read_custom_dataset("Dataset of Mammography with Benign Malignant Breast Masses/INbreast+MIAS+DDSM Dataset (1)/","breast","binary-class",18,19,10)
  #read_custom_dataset("chest_data/classifier_data/","chestdata","binary-class",18,19,10)
  #read_dataset('data/chestmnist.npz','chestmnist','multi-label binary-class',10,20,10)
  #read_dataset('data/dermamnist.npz','dermamnist','multi-label binary-class',10,20,10)
  #read_dataset('data/octmnist.npz','octmnist','multi-label binary-class',10,20,10)
  #read_dataset('data/pneumoniamnist.npz','pneumoniamnist','binary-class',10,20,10)
  #read_dataset('data/retinamnist.npz','retinamnist','multi-label binary-class',10,20,10)
  #read_dataset('data/breastmnist.npz','breastmnist','binary-class',10,20,10)
  #read_dataset('data/bloodmnist.npz','bloodmnist','multi-label binary-class',10,20,10)
  #read_dataset('data/tissuemnist.npz','tissuemnist','multi-label binary-class',2,1,1)
  #read_dataset('data/organamnist.npz','organamnist','multi-label binary-class',10,20,10)
  #read_dataset('data/organcmnist.npz','organcmnist','multi-label binary-class',10,20,10)
  #read_dataset('data/organsmnist.npz','organcmnist','multi-label binary-class',10,20,10)


  #read_dataset('data/nodulemnist3d.npz','nodulemnist3d','binary-class',10,20,10)
  #read_dataset('data/organmnist3d.npz','organmnist3d','multi-label binary-class',10,20,10)
  #read_dataset('data/adrenalmnist3d.npz','adrenalmnist3d','binary-class',10,20,10)
  #read_dataset("data/fracturemnist3d.npz","fracturemnist3d","multi-label binary-class",18,19,1)
  #read_dataset("data/synapsemnist3d.npz","synapsemnist3dturemnist3d","binary-class",18,19,10)
  #read_dataset('data/vesselmnist3d.npz','vesselmnist3d','binary-class',2,1,1)