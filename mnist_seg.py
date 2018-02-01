"""
Author: Héctor Sánchez
Date: January-30-2018
Description: MNIST segmentation
"""

# TODO: 
"""
Dropout not working with tensorboard summary histogram.

InvalidArgumentError (see above for traceback): Nan in summary histogram for: conv2/activations
         [[Node: conv2/activations = HistogramSummary[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"](conv2/activations/tag, conv2/Relu)]]
"""

import os
import argparse
import utils as ut

desc_msg = 'MNIST segmentation using tensorflow and some sort of LeNet-5'
parser = argparse.ArgumentParser(desc_msg)

### START: Select the device where the operations will be executed

# All posible choices for device selection
from tensorflow.python.client import device_lib

with ut.HiddenPrints():
  all_devices = device_lib.list_local_devices()

choices = []
for i in range(len(all_devices)):
  choices.append(i)

device_default = choices[-1]

msg = 'Select which device tf should use for the ops.'\
			+' 0: CPU, 1>=GPU (if available).'
parser.add_argument('-d','--device',help=msg,type=int,
                    choices=choices,default=device_default)



######## STARTS: Other args
msg = 'Select which model to train/predict:\n\t'
msg += '0 - Strides without max pooling\n\t'
msg += '1 - Max pooling for dimensional reduction'
parser.add_argument('-m','--model',help=msg,
      type=int,default=0,choices=[0,1])
parser.add_argument('--do',help='Use DropOut if selected',
      action='store_true')
parser.add_argument('--dop',help='DropOut probability',
      type=float,default=0.25)
parser.add_argument('--lr',help='Define a different learning rate',
      type=float,default=3e-7)
parser.add_argument('-i','--iterations',help='Number of training it.',
      type=int,default=10000)
parser.add_argument('-s','--save',help='Saves checkpoints of the model',
      action='store_true')
parser.add_argument('-l','--load',help='Load model from checkpoint',
      action='store_true')
parser.add_argument('--step',help='Step to be loaded from checkpoint',
      type=int)
parser.add_argument('--tb_log',help='Saves a log of the training process',
      action='store_true')
######## ENDS: Other args



args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

msg = '\n\n################\n\n'
msg = msg + '-Device used for thensorflow: {0}'.format(\
      all_devices[args.device].name)\
      + msg
print(msg)

### END: Select the device where the operations will be executed

from datetime import datetime as dt
import tensorflow as tf
import numpy as np

from MNIST import load_mnist
import models

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
  np.random.shuffle(ind)

  ind = int(ind.shape[0]*0.9)
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

  print('\tTrain data: {0} - {1}'.format(train.images.shape,train.cls.shape))
  print('\tVal data: {0} - {1}'.format(val.images.shape,val.cls.shape))
  print('\tTest data: {0} - {1}'.format(test.images.shape,test.cls.shape))

  
  model = models.SegModel(train=train,val=val,test=test,model=args.model,
                save=args.save,load=args.load,load_step=args.step,
                lr=args.lr,dropout=args.do,drop_prob=args.dop,
                tb_log=args.tb_log)
  
  model.train(num_it=args.iterations,verb=100)
    