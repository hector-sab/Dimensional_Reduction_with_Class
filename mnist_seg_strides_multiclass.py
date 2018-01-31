"""
Author: Héctor Sánchez
Date: January-30-2018
Description: MNIST segmentation
"""

import argparse
desc_msg = 'MNIST segmentation using tensorflow and some sort of LeNet-5'
parser = argparse.ArgumentParser(desc_msg)

### START: Select the device where the operations will be executed

# All posible choices for device selection
from tensorflow.python.client import device_lib
all_devices = device_lib.list_local_devices()

choices = []
for i in range(len(all_devices)):
  choices.append(i)

parser.add_argument('-d','--device',help='Select which device ',type=int,
                    choices=choices,default=0)

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

### END: Select the device where the operations will be executed

import os
from datetime import datetime as dt


if __name__=='__main__':
	pass