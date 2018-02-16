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
  """
  Description:
  Main model for MNIST segmentation. Do not use by itself. Use it as superclass
  to inherit attributes
  """
  def __init__(self,inp,num_class=11,version=1,histogram=True,
    dropout=False,drop_prob=0.85,def_cp_name='mnist_seg',
    def_cp_path='checkpoints/mnist_seg',def_log_name='mnist_seg',
    def_log_path='./log/'):
    """
    inp: Input placeholder.
    shape: Tensorflow tensor shape used in the input placeholder.
           It must be a list object.
    dropout: Flag used to indicate if dropout will be used
    drop_prob: Percentage of neurons to be turned off
    histogram: Indicates if information for tensorboard should be annexed.
    """
    """
    if def_cp_name is None:
      def_cp_name = 'mnist_seg'
    if def_cp_path is None:
      def_cp_path = 'checkpoints/mnist_seg'
    if def_log_name is None:
      def_log_name = 'mnist_seg'
    if def_log_path is None:
      def_log_path = './log/'
    """
    ####-S: Naming of generated files 
    self.def_cp_name = def_cp_name
    self.def_cp_path = ut.vali_end_path(def_cp_path+'v'+str(version))
    self.def_log_name = def_log_name+'v'+str(version)
    self.def_log_path = ut.vali_end_path(def_log_path)
    ####-E: Naming of generated files
    ##
    ####-S: Network variables
    self.x = inp
    self.dropout = dropout
    self.drop_prob = drop_prob
    self.reg = [] # Contains l2 regularizaton for weights
    self.histogram = histogram
    self.convs = []
    self.deconvs = []
    ####-E: Network variables
    ##
    ####-S: Data specs
    self.im_h = int(self.x.get_shape()[1])
    self.im_w = int(self.x.get_shape()[2])
    self.im_c = int(self.x.get_shape()[3])
    self.num_class = num_class
    ####-E: Data specs

  def get_inp(self):
    return(self.x)
  
  def last_layer(self):
    """
    Returns the last layer of the model
    """
    return(self.deconvs[-1])

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

  def log_dir(self,path,name):
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
  """
  Description:
  Model for MNIST segmentation using max pooling but no skip connections
  """
  def __init__(self,inp,num_class=11,version=1,histogram=False,
    dropout=False,drop_prob=0.85,def_cp_name='mnist_seg',
    def_cp_path='checkpoints/mnist_seg_mpnosc',def_log_name='mnist_seg_mpnosc',
    def_log_path='./log/',ex=1):
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

    self.ex = ex#tf.placeholder_with_default(1,shape=[])
    
    ####-S: Core Model
    self.model = self.core_model()
    ####-E: Core Model

  def core_model(self):
    ####-S: Network Specs
    # Each position represents the convolution to which it belogs
    cks = [3,3,3,3,3,3]
    cnum_k = [self.im_c,16,16,32,32,64,64]
    dks = [3,3,3,3,3,3]
    dnum_k = [64,64,32,32,16,16,self.num_class]
    ####-E: Network Specs
    ##
    ####-S: Core Model
    self.pools = []
    self.ind = []

    shape = [cks[0],cks[0],cnum_k[0],cnum_k[1]]
    conv,reg = tut.conv(inp=self.x,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv1')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\n\t{}'.format(conv))

    shape = [cks[1],cks[1],cnum_k[1],cnum_k[2]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv2')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    pool,ind = tut.max_pool(conv,args=True,name='maxpool1')
    self.pools.append(pool)
    self.ind.append(ind)
    print('\t{}'.format(pool))

    shape = [cks[2],cks[2],cnum_k[2],cnum_k[3]]
    conv,reg = tut.conv(inp=pool,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv3')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    shape = [cks[3],cks[3],cnum_k[3],cnum_k[4]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv4')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    pool,ind = tut.max_pool(conv,args=True,name='maxpool2')
    self.pools.append(pool)
    self.ind.append(ind)
    print('\t{}'.format(pool))

    shape = [cks[4],cks[4],cnum_k[4],cnum_k[5]]
    conv,reg = tut.conv(inp=pool,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv5')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    shape = [cks[5],cks[5],cnum_k[5],cnum_k[6]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv6')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))



    shape = [dks[0],dks[0],dnum_k[1],dnum_k[0]]
    deconv,reg = tut.deconv(inp=conv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv1')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\n\t{}'.format(deconv))

    shape = [dks[1],dks[1],dnum_k[2],dnum_k[1]]
    deconv,reg = tut.deconv(inp=deconv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv2')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    unpool = tut.unpool_with_argmax(deconv,ind=self.ind[1],
                    input_shape=[self.ex,deconv.get_shape()[1].value,
                                  deconv.get_shape()[2].value,dnum_k[2]],
                    name='unpool1')
    print('\t{}'.format(unpool))

    shape = [dks[2],dks[2],dnum_k[3],dnum_k[2]]
    deconv,reg = tut.deconv(inp=unpool,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv3')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    shape = [dks[3],dks[3],dnum_k[4],dnum_k[3]]
    deconv,reg = tut.deconv(inp=deconv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv4')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    unpool = tut.unpool_with_argmax(deconv,ind=self.ind[0],
                    input_shape=[self.ex,deconv.get_shape()[1].value,
                                  deconv.get_shape()[2].value,dnum_k[4]],
                    name='unpool2')
    print('\t{}'.format(unpool))

    shape = [dks[4],dks[4],dnum_k[5],dnum_k[4]]
    deconv,reg = tut.deconv(inp=unpool,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv5')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    shape = [dks[5],dks[5],dnum_k[6],dnum_k[5]]
    deconv,reg = tut.deconv(inp=deconv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv6')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))
    ####-E: Core Model

class MaxPoolSC(Model):
  """
  Description:
  Model for MNIST segmentation using max pooling and skip connections
  """
  def __init__(self,inp,num_class=11,version=1,histogram=False,
    dropout=False,drop_prob=0.85,def_cp_name='mnist_seg',
    def_cp_path='checkpoints/mnist_seg_mpsc',def_log_name='mnist_seg_mpsc',
    def_log_path='./log/',ex=1):
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

    self.ex = ex #tf.placeholder_with_default(1,shape=[])
    
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
    self.pools = []
    self.ind = []

    shape = [cks[0],cks[0],cnum_k[0],cnum_k[1]]
    conv,reg = tut.conv(inp=self.x,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv1')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\n\t{}'.format(conv))

    shape = [cks[1],cks[1],cnum_k[1],cnum_k[2]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv2')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    pool,ind = tut.max_pool(conv,args=True,name='maxpool1')
    self.pools.append(pool)
    self.ind.append(ind)
    print('\t{}'.format(pool))

    shape = [cks[2],cks[2],cnum_k[2],cnum_k[3]]
    conv,reg = tut.conv(inp=pool,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv3')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    shape = [cks[3],cks[3],cnum_k[3],cnum_k[4]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv4')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    pool,ind = tut.max_pool(conv,args=True,name='maxpool2')
    self.pools.append(pool)
    self.ind.append(ind)
    print('\t{}'.format(pool))

    shape = [cks[4],cks[4],cnum_k[4],cnum_k[5]]
    conv,reg = tut.conv(inp=pool,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv5')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    shape = [cks[5],cks[5],cnum_k[5],cnum_k[6]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv6')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))



    shape = [dks[0],dks[0],dnum_k[1],dnum_k[0]]
    deconv,reg = tut.deconv(inp=conv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv1')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\n\t{}'.format(deconv))

    shape = [dks[1],dks[1],dnum_k[2],dnum_k[1]]
    deconv,reg = tut.deconv(inp=deconv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv2')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    unpool = tut.unpool_with_argmax(deconv,ind=self.ind[1],
                    input_shape=[self.ex,deconv.get_shape()[1].value,
                                  deconv.get_shape()[2].value,dnum_k[2]],
                    name='unpool1')
    print('\t{}'.format(unpool))

    sc_sum = self.convs[3] + unpool

    shape = [dks[2],dks[2],dnum_k[3],dnum_k[2]]
    deconv,reg = tut.deconv(inp=sc_sum,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv3')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    shape = [dks[3],dks[3],dnum_k[4],dnum_k[3]]
    deconv,reg = tut.deconv(inp=deconv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv4')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    unpool = tut.unpool_with_argmax(deconv,ind=self.ind[0],
                    input_shape=[self.ex,deconv.get_shape()[1].value,
                                  deconv.get_shape()[2].value,dnum_k[4]],
                    name='unpool2')
    print('\t{}'.format(unpool))

    sc_sum = self.convs[1] + unpool

    shape = [dks[4],dks[4],dnum_k[5],dnum_k[4]]
    deconv,reg = tut.deconv(inp=sc_sum,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv5')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    shape = [dks[5],dks[5],dnum_k[6],dnum_k[5]]
    deconv,reg = tut.deconv(inp=deconv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv6')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))
    ####-E: Core Model

class StrideSC(Model):
  """
  Description:
  Model for MNIST segmentation using striding without skip connections
  """
  def __init__(self,inp,num_class=11,version=1,histogram=False,
    dropout=False,drop_prob=0.85,def_cp_name='mnist_seg',
    def_cp_path='checkpoints/mnist_seg_stsc',def_log_name='mnist_seg_stsc',
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
    cnum_k = [self.im_c,16,16,32,32,64,64]
    dks = [3,3,3,3,3,3]
    dnum_k = [64,64,32,32,16,16,self.num_class]
    ####-E: Network Specs
    ##
    ####-S: Core Model
    shape = [cks[0],cks[0],cnum_k[0],cnum_k[1]]
    conv,reg = tut.conv(inp=self.x,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv1')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\n\t{}'.format(conv))

    shape = [cks[1],cks[1],cnum_k[1],cnum_k[2]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,strides=[1,2,2,1],name='conv2')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    shape = [cks[2],cks[2],cnum_k[2],cnum_k[3]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv3')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    shape = [cks[3],cks[3],cnum_k[3],cnum_k[4]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,strides=[1,2,2,1],name='conv4')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    shape = [cks[4],cks[4],cnum_k[4],cnum_k[5]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv5')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    shape = [cks[5],cks[5],cnum_k[5],cnum_k[6]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv6')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))



    shape = [dks[0],dks[0],dnum_k[1],dnum_k[0]]
    deconv,reg = tut.deconv(inp=conv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv1')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\n\t{}'.format(deconv))

    shape = [dks[1],dks[1],dnum_k[2],dnum_k[1]]
    deconv,reg = tut.deconv(inp=deconv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv2')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    sc_sum = self.convs[3] + deconv

    shape = [dks[2],dks[2],dnum_k[3],dnum_k[2]]
    deconv,reg = tut.deconv(inp=sc_sum,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv3')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    shape = [dks[3],dks[3],dnum_k[4],dnum_k[3]]
    deconv,reg = tut.deconv(inp=deconv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,strides=[1,2,2,1],name='deconv4')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    sc_sum = self.convs[1] + deconv

    shape = [dks[4],dks[4],dnum_k[5],dnum_k[4]]
    deconv,reg = tut.deconv(inp=sc_sum,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv5')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    shape = [dks[5],dks[5],dnum_k[6],dnum_k[5]]
    deconv,reg = tut.deconv(inp=deconv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,strides=[1,2,2,1],name='deconv6')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))
    ####-E: Core Model

class StrideNoSC(Model):
  """
  Description:
  Model for MNIST segmentation using striding with skip connections
  """
  def __init__(self,inp,num_class=11,version=1,histogram=False,
    dropout=False,drop_prob=0.85,def_cp_name='mnist_seg',
    def_cp_path='checkpoints/mnist_seg_stnosc',def_log_name='mnist_seg_stnosc',
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
    cnum_k = [self.im_c,16,16,32,32,64,64]
    dks = [3,3,3,3,3,3]
    dnum_k = [64,64,32,32,16,16,self.num_class]
    ####-E: Network Specs
    ##
    ####-S: Core Model
    shape = [cks[0],cks[0],cnum_k[0],cnum_k[1]]
    conv,reg = tut.conv(inp=self.x,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv1')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\n\t{}'.format(conv))

    shape = [cks[1],cks[1],cnum_k[1],cnum_k[2]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,strides=[1,2,2,1],name='conv2')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    shape = [cks[2],cks[2],cnum_k[2],cnum_k[3]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv3')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    shape = [cks[3],cks[3],cnum_k[3],cnum_k[4]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,strides=[1,2,2,1],name='conv4')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    shape = [cks[4],cks[4],cnum_k[4],cnum_k[5]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv5')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))

    shape = [cks[5],cks[5],cnum_k[5],cnum_k[6]]
    conv,reg = tut.conv(inp=conv,shape=shape,histogram=self.histogram,
              l2=True,relu=True,name='conv6')
    self.convs.append(conv)
    self.reg.append(reg)
    print('\t{}'.format(conv))



    shape = [dks[0],dks[0],dnum_k[1],dnum_k[0]]
    deconv,reg = tut.deconv(inp=conv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv1')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\n\t{}'.format(deconv))

    shape = [dks[1],dks[1],dnum_k[2],dnum_k[1]]
    deconv,reg = tut.deconv(inp=deconv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv2')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    shape = [dks[2],dks[2],dnum_k[3],dnum_k[2]]
    deconv,reg = tut.deconv(inp=deconv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv3')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    shape = [dks[3],dks[3],dnum_k[4],dnum_k[3]]
    deconv,reg = tut.deconv(inp=deconv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,strides=[1,2,2,1],name='deconv4')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    shape = [dks[4],dks[4],dnum_k[5],dnum_k[4]]
    deconv,reg = tut.deconv(inp=deconv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,name='deconv5')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))

    shape = [dks[5],dks[5],dnum_k[6],dnum_k[5]]
    deconv,reg = tut.deconv(inp=deconv,shape=shape,histogram=self.histogram,
                  l2=True,relu=True,strides=[1,2,2,1],name='deconv6')
    self.deconvs.append(deconv)
    self.reg.append(reg)
    print('\t{}'.format(deconv))
    ####-E: Core Model


class AtrousMPNoSC(Model):
  def __init__(self,inp,num_class=11,version=1,histogram=False,
    dropout=False,drop_prob=0.85,def_cp_name='mnist_seg',
    def_cp_path='checkpoints/mnist_seg_mpsconeo',
    def_log_name='mnist_seg_mpsconeo',def_log_path='./log/'):
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
    cnum_k = [self.im_c,16,16,32,32,64,64]
    dks = [3,3,3,3,3,3]
    dnum_k = [64,64,32,32,16,16,self.num_class]
    ####-E: Network Specs
    ##
    ####-S: Core Model

    ####-E: Core Model
