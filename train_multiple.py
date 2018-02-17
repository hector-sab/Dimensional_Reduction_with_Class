# coding=utf-8
"""
Author: Héctor Sánchez
Date: January-30-2018
Description: MNIST segmentation
  Use this script to train several models.

  $ python train_multiple.py -d 0
"""
import os
import argparse
import utils as ut
import tensorflow as tf

desc_msg = 'MNIST segmentation using tensorflow and some sort of LeNet-5'
parser = argparse.ArgumentParser(desc_msg)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#from tensorflow.python.client import device_lib

#with ut.HiddenPrints():
#  all_devices = device_lib.list_local_devices()
all_devices = [0,1,2,3,4]
choices = []
for i in range(len(all_devices)):
  choices.append(i)

device_default = choices[-1]

msg = 'Select which device tf should use for the ops.'\
      +' 0: CPU, 1>=GPU (if available).'
parser.add_argument('-d','--device',help=msg,type=int,
                    choices=choices,default=device_default)

msg = 'Select which models to train:\n\t'
msg += '0 - Max pooling for dimensional reduction\n\t'
msg += '1 - Strides without max pooling'
parser.add_argument('-m','--model',help=msg,
      type=int,default=0,choices=[0,1,2])
parser.add_argument('--lr',help='Define a different learning rate',
      type=float,default=3e-9)
parser.add_argument('-i','--iterations',help='Number of training it.',
      type=int,default=1000000)
parser.add_argument('--bs',help='Size of batch for training',
      type=int,default=20)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

msg = '\n\n################\n\n'
#msg = msg + '-Device used for thensorflow: {0}'.format(\
#      all_devices[args.device].name)\
#      + msg
print(msg)

import numpy as np
from MNIST import load_mnist
import seg_model as models


if __name__=='__main__':
  # Load MNIST data
  msg = 'Loading MNIST data...\n'
  print(msg)
  MNIST_path = './MNIST/'
  MNIST_files = ['train-images.idx3-ubyte','train-labels.idx1-ubyte',
                 't10k-images.idx3-ubyte','t10k-labels.idx1-ubyte']
  train_ims = load_mnist(MNIST_path+MNIST_files[0])
  train_cls  = load_mnist(MNIST_path+MNIST_files[1])
  test_ims = load_mnist(MNIST_path+MNIST_files[2])
  test_cls= load_mnist(MNIST_path+MNIST_files[3])

  # Fix ims and cls shape
  train_ims = train_ims.reshape(-1,train_ims.shape[1],train_ims.shape[2],1)
  train_cls = train_cls.reshape(-1,1)
  test_ims = test_ims.reshape(-1,test_ims.shape[1],test_ims.shape[2],1)
  test_cls = test_cls.reshape(-1,1)

  # Split train into train and validation

  ind = np.arange(train_ims.shape[0])
  np.random.seed(3) # Ensures repeatability
  np.random.shuffle(ind)

  ind = int(ind.shape[0]*0.95)
  val_ims = train_ims[ind:]
  val_cls = train_cls[ind:]
  train_ims = train_ims[:ind]
  train_cls = train_cls[:ind]

  train = ut.DataSeg(ims=train_ims,cls=train_cls)
  val = ut.DataSeg(ims=val_ims,cls=val_cls)
  test = ut.DataSeg(ims=test_ims,cls=test_cls)

  # Frees memory
  train_ims = None
  train_cls = None
  val_ims = None
  val_ims = None
  test_ims = None
  test_cls = None

  print('Train data: {0} - {1}'.format(train.images.shape,train.cls.shape))
  print('Val data: {0} - {1}'.format(val.images.shape,val.cls.shape))
  print('Test data: {0} - {1}'.format(test.images.shape,test.cls.shape))

  bs = args.bs
  lr = args.lr

  if args.model==0:
    for i in [0,1]:
      tf.reset_default_graph()
      print('\n-----> Executing model {}'.format(i))
      model = models.SegModel(train=train,val=val,test=test,model=i,training=True,
                bs=bs,save=True,load=False,lr=lr,tb_log=True,ex=bs,max_to_keep=50000,
                version=1,histogram=True)
  
      model.optimize(num_it=args.iterations,verb=100)
      model.close_session()
      print('Done with model: {0}'.format(i))
  elif args.model==1:
    for i in [2,3]:
      tf.reset_default_graph()
      print('\n-----> Executing model {}'.format(i))
      model = models.SegModel(train=train,val=val,test=test,model=i,training=True,
                bs=bs,save=True,load=False,lr=lr,tb_log=True,max_to_keep=50000,
                version=1,histogram=True)
  
      model.optimize(num_it=args.iterations,verb=100)
      model.close_session()
      print('Done with model: {0}'.format(i))
  elif args.model==2:
    for i in [4,5]:
      tf.reset_default_graph()
      print('\n-----> Executing model {}'.format(i))
      model = models.SegModel(train=train,val=val,test=test,model=i,training=True,
                bs=bs,save=True,load=False,lr=lr,tb_log=True,max_to_keep=50000,
                version=1,histogram=True)

      model.optimize(num_it=args.iterations,verb=100)
      model.close_session()
      print('Done with model: {0}'.format(i))
  """
  elif args.model==2:
    for i in [3]:
      tf.reset_default_graph()
      print('\n-----> Executing model {}'.format(i))
      model = models.SegModelSigmoid(train=train,val=val,test=test,model=i,training=True,
                bs=bs,save=True,load=False,lr=3e-7,tb_log=True,ex=bs,max_to_keep=50000,
                version=2,histogram=True)
  
      model.optimize(num_it=args.iterations,verb=100)
      model.close_session()
      print('Done with model: {0}'.format(i))
  """