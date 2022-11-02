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

def read_dataset(file_path,task_name,task_type,pop_size,max_pop_size,epochs):

  data = load(file_path)
  x_train,y_train,x_val,y_val,x_test,y_test = data['train_images'],data['train_labels'],data['val_images'],data['val_labels'],data['test_images'],data['test_labels']

  data = {'x_train': x_train, 'y_train': y_train,'x_val':x_val,'y_val':y_val, 'x_test': x_test, 'y_test': y_test}
  a = AutoCNN(pop_size, max_pop_size,task_type,data,task_name, epoch_number=15)
  a.run()
  


if __name__ == '__main__':
  #read_dataset("data/pathmnist.npz","pathmnist","multi-label binary-class",18,19,10)
  #read_dataset('data/chestmnist.npz','chestmnist','multi-label binary-class',10,20,10)
  #read_dataset('data/dermamnist.npz','dermamnist','multi-label binary-class',10,20,10)
  #read_dataset('data/octmnist.npz','octmnist','multi-label binary-class',22,20,10)
  #read_dataset('data/pneumoniamnist.npz','pneumoniamnist','binary-class',10,20,10)
  #read_dataset('data/retinamnist.npz','retinamnist','multi-label binary-class',10,20,10)
  #read_dataset('data/breastmnist.npz','breastmnist','binary-class',10,20,10)
  #read_dataset('data/bloodmnist.npz','bloodmnist','multi-label binary-class',10,20,10)
  #read_dataset('data/tissuemnist.npz','tissuemnist','multi-label binary-class',22,20,10)
  read_dataset('data/organamnist.npz','organamnist','multi-label binary-class',22,30,15)
  #read_dataset('data/organcmnist.npz','organcmnist','multi-label binary-class',18,19,10)
  #read_dataset('data/organsmnist.npz','organcmnist','multi-label binary-class',18,19,10)


  #read_dataset('data/nodulemnist3d.npz','nodulemnist3d','binary-class',18,19,10)
  #read_dataset('data/organmnist3d.npz','organmnist3d','multi-label binary-class',18,19,10)
  #read_dataset('data/adrenalmnist3d.npz','adrenalmnist3d','binary-class',10,20,10)
  #read_dataset("data/fracturemnist3d.npz","fracturemnist3d","multi-label binary-class",18,19,10)
  #read_dataset("data/synapsemnist3d.npz","synapsemnist3d","binary-class",18,19,10)
  #read_dataset('data/vesselmnist3d.npz','vesselmnist3d','binary-class',18,19,10)