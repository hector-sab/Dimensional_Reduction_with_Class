# coding=utf-8
"""
Author: Héctor Sánchez
Date: January-31-2018
Description: Contains the different models for MNIST segmentation
"""
import os
# Hidde debug infro from tf... https://stackoverflow.com/a/38645250/5969548
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import utils as ut
import tfutils as tut

class Model:
  def __init__(self,inp,num_class=11,version=1,histogram=True,
    dropout=False,drop_prob=0.85,def_cp_name=None,
    def_cp_path=None,def_log_name=None,
    def_log_path=None):
    """
    inp: Input placeholder.
    shape: Tensorflow tensor shape used in the input placeholder.
           It must be a list object.
    dropout: Flag used to indicate if dropout will be used
    drop_prob: Percentage of neurons to be turned off
    histogram: Indicates if information for tensorboard should be annexed.
    """
    if def_cp_name is None:
      def_cp_name = 'mnist_seg'
    if def_cp_path is None:
      def_cp_path = 'checkpoints/mnist_seg'
    if def_log_name is None:
      def_log_name = 'mnist_seg'
    if def_log_path is None:
      def_log_path = './log/'

    ####-S: Naming of generated files 
    self.def_cp_name = def_cp_name
    self.def_cp_path = ut.vali_end_path(def_cp_path+'v'+str(version))
    self.def_log_name = def_log_name+'v'+str(version)
    self.def_log_path = ut.vali_end_path(def_log_path)
    ####-E: Naming of generated files
    ##
    ####-S: Data specs
    self.im_h = int(self.x.get_shape()[1])
    self.im_w = int(self.x.get_shape()[2])
    self.im_c = int(self.x.get_shape()[3])
    self.num_class = num_class
    ####-E: Data specs
    ##
    ####-S: Network variables
    self.x = inp
    self.dropout = dropout
    self.drop_prob = drop_prob
    self.reg = [] # Contains l2 regularizaton for weights
    ####-E: Network variables
  
  def last_layer(self):
    """
    Returns the last layer of the model
    """
    return(self.deconv[-1])

  def checkpoint_dir(self,path,name):
    """
    Indicates where to save/load the model
    """
    def_name = self.def_cp_name
    def_path = self.def_cp_path

    if path is None and name is None:
      path = def_path
      name = def_name
    elif path is None and name is not None:
      path = def_path
    elif path is not None and name is None:
      name = def_name

    return(path,name)

    """
    Indicates where to save log used for tensorboard
    """
    def_name = self.def_log_name
    def_path = self.def_log_path

    if path is None and name is None:
      path = def_path
      name = def_name
    elif path is None and name is not None:
      path = def_path
    elif path is not None and name is None:
      name = def_name

    return(path,name)

class MaxPoolNoSC(Model):
  def __init__(self,inp,num_class=11,version=1,histogram=True,
    dropout=False,drop_prob=0.85,def_cp_name='mnist_seg',
    def_cp_path='checkpoints/mnist_seg_mp',def_log_name='mnist_seg_mp',
    def_log_path='./log/'):
    """
    inp: Input placeholder.
    shape: Tensorflow tensor shape used in the input placeholder.
           It must be a list object.
    dropout: Flag used to indicate if dropout will be used
    drop_prob: Percentage of neurons to be turned off
    histogram: Indicates if information for tensorboard should be annexed.
    """
    Model.__init__(self,inp,num_class,version,histogram,dropout,
      drop_prob,def_cp_name,def_cp_path,def_log_name,def_log_path)
    ####-S: Core Model
    self.model = self.core_model()
    ####-E: Core Model

  def core_model(self):
    ####-S: Network Specs
    # Each position represents the convolution to which it belogs
    cks = [3,3,3,3,3,3]
    cnum_k = [self.im_c,8,8,16,16,32,32]
    dks = [3,3,3,3,3,3]
    dnum_k = [32,32,16,16,8,8,self.num_class]
    ####-E: Network Specs
    ##
    ####-S: Core Model
    self.convs = []
    self.ind = []
    count = 0
    for i in enumerate(len(cks)):
      if i==0:
        input_ = self.x
      else:
        input_ = self.convs[i-1]

      shape = [cks[i],ks[i],cnum_k[i],cnum_k[i+1]]
      conv,reg = tut.conv(inp=input_,shape=shape,histogram=histogram,
                    l2=True,relu=True,name='conv'+str(i))
      print(conv)

      self.convs.append(conv)
      self.reg.append(reg)
      if i+1%2==0 and i<5:
        pool,ind = tut.max_pool(conv,args=True)
        print(pool)
        self.convs.append(pool)
        self.ind.append(ind)
        count += 1


    self.deconvs = []
    
    for i in enumerate(len(dks)):
      if i+1%2==0 and i<5:
        input_ = tut.unpool_with_argmax(self.deconvs[i-1],self.ind[count-1],
                    name='unpool'+str(count-1))
        print(input_)
        self.deconvs.append(input_)
        count -= 1
      elif i==0:
        input_ = self.convs[-1]
      else:
        input_ = self.deconvs[i-1]

      shape = [dks[i],dks[i],dnum_k[i+1],dnum_k[i]]
      deconv,reg = tut.deconv(inp=input_,shape=shape,histogram=histogram,
                      l2_reg=True,relu=True,name='deconv'+str(i))
      print(deconv)

      self.deconvs.append(deconv)
      self.reg.append(reg)
    ####-E: Core Model
