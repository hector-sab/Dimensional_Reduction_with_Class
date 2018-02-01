import os
import sys
import numpy as np
# Hidde debug infro from tf... https://stackoverflow.com/a/38645250/5969548
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import matplotlib.pyplot as plt

class DataSeg:
  """
  Used for MNIST-like segmentation
  """
  def __init__(self,ims,cls,num_cls=10):
    self.images = ims
    self.cls = cls 

    self.bt_init = 0

  def im2seg(self,init,end):
    """
    Returns the a segmented image where 0 means background
    and any other number represents the other classes
    """
    ims_rs = self.images[init:end]
    shape = ims_rs.shape
    ims_rs = ims_rs.reshape(shape[0],-1)
    data1 = (ims_rs==0)*-1
    data2 = (ims_rs!=0)*self.cls[init:end]
    data = data1 + data2
    data = data + 1
    data = data.reshape(shape)
    return(data)

  def im2seg2(self,init,end):
    """
    Returns the a segmented image where 0 means background
    and one represents foreground
    """
    data = (self.images[init:end]!=0)*1
    return(data)

  def restart_next_batch(self):
    self.bt_init = 0

  def next_batch(self,bs,seg_fb=False,res_count=False):
    """
    seg_fb: Foreground/Background only segmentation if True
            if False, it will segment by classes
    res_count: It will restar the count of the batch feed
    """
    end_batch = self.bt_init + bs
    new_ep = False
    
    if end_batch>= self.images.shape[0] and (self.images.shape[0]-1-self.bt_init)>0:
      end_batch = self.images.shape[0]-1
    elif end_batch>=self.images.shape[0]:
      self.bt_init = 0
      end_batch = self.bt_init + bs
      new_ep = True
    
    ims = self.images[self.bt_init:end_batch]
    cls = self.cls[self.bt_init:end_batch]
    if not seg_fb:
      seg = self.im2seg(self.bt_init,end_batch)
    else:
      seg = self.im2seg2(self.bt_init,end_batch)

    self.bt_init = end_batch
    
    # Total number of batches
    num_bt = int(self.images.shape[0]/bs)
    
    out = {'ims': ims, 'seg': seg,'cls':cls, 'new_ep': new_ep, 'num_batches': num_bt}
    
    return(out)

class Data:
  def __init__(self,ims,cls,num_cls=10):
    """
    self.cls: [num_ims]
    self.labels: [num_ims,num_cls]
    """
    self.images = ims
    self.cls = cls
    self.labels = self.cls2onehot(num_cls)
    
    self.bt_init = 0
    
  def cls2onehot_seg(self,num_cls):
    """
    TODO: Fix it
    """
    num_ims = self.images.shape[0]
    onehot = np.zeros((num_ims,num_cls))
    onehot[np.arange(num_ims),self.cls] = 1
    return(onehot)

  def cls2onehot(self,num_cls):
    x = self.cls.shape[0]
    onehot = np.zeros((x,num_cls))
    ind = np.arange(x)
    onehot[ind,self.cls.flat] = 1
    return(onehot)

  def next_batch(self,bs):
    end_batch = self.bt_init + bs
    new_ep = False
    
    if end_batch>= self.images.shape[0] and (self.images.shape[0]-1-self.bt_init)>0:
      end_batch= self.images.shape[0]-1
    elif end_batch>=self.images.shape[0]:
      self.bt_init = 0
      end_batch = self.bt_init + bs
      new_ep = True
    
    ims = self.images[self.bt_init:end_batch]
    cls = self.cls[self.bt_init:end_batch]
    lbs = self.labels[self.bt_init:end_batch]
    
    self.bt_init = end_batch
    
    # Total number of batches
    num_bt = int(self.images.shape[0]/bs)
    
    out = {'ims': ims,'cls':cls, 'lbs': lbs, 'new_ep': new_ep, 'num_batches': num_bt}
    
    return(out)

class HiddenPrints:
  """
  Description: Hides all print functions from a function
  
  Taken from:
  https://stackoverflow.com/a/45669280/5969548

  How to use it...

  with HiddenPrints():
    print("This will not be printed") # This won't print
  """
  def __enter__(self):
    self._original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

  def __exit__(self, exc_type, exc_val, exc_tb):
    sys.stdout = self._original_stdout

def plot_ims(ims,cls_true=None,cls_pred=None,nrows=3,ncols=3):
  """
  Plot the images with numeric labels
  """
  fig,axes = plt.subplots(nrows=nrows,ncols=ncols)
  fig.subplots_adjust(hspace=0.3,wspace=0.3)
  

  for i,im in enumerate(ims):
    axes.flat[i].imshow(im.reshape(im.shape[0],im.shape[1]),cmap='binary')
    
    # Remove ticks from the plot
    axes.flat[i].set_xticks([])
    axes.flat[i].set_yticks([])
    
    if nrows==1 and ncols==1:
      axes.imshow(ims.reshape(ims.shape[0],ims.shape[1]),cmap='binary')
      # Remove ticks from the plot
      axes.set_xticks([])
      axes.set_yticks([])

      if cls_true!=None and cls_pred!=None:
        true = cls_true
        pred = cls_pred
        xlabel = 'True: {0}, Pred: {1}'.format(true,pred)
      elif cls_true!=None:
        true = cls_true
        xlabel = 'True: {0}'.format(true)
      elif cls_pred!=None:
        pred = cls_pred
        xlabel = 'Pred: {0}'.format(pred)
      else:
          xlabel = ''

      axes.set_xlabel(xlabel)
    else:
      if isinstance(cls_true,np.ndarray) and isinstance(cls_pred,np.ndarray):
        true = cls_true[i][0]
        pred = cls_pred[i][0]
        xlabel = 'True: {0}, Pred: {1}'.format(true,pred)
      elif isinstance(cls_true,np.ndarray):
        true = cls_true[i][0]
        xlabel = 'True: {0}'.format(true)
      elif isinstance(cls_pred,np.ndarray):
        pred = cls_pred[i][0]
        xlabel = 'Pred: {0}'.format(pred)
      else:
        xlabel = ''
      
      axes.flat[i].set_xlabel(xlabel)
  plt.show()

def plot_ims_let(ims,cls_true=None,cls_pred=None,nrows=3,ncols=3):
  """
  Plots images with the letters as their labels
  """
  letters = np.array(['-','A','B','C','D','E','F','G','H','I',
                      'J','K','L','M','N','O','P','Q',
                      'R','S','T','U','V','W','X','Y','Z'])
  fig,axes = plt.subplots(nrows=nrows,ncols=ncols)
  fig.subplots_adjust(hspace=0.3,wspace=0.3)
  
  if nrows==1 and ncols==1:
    axes.imshow(ims.reshape(ims.shape[0],ims.shape[1]),cmap='binary')
    # Remove ticks from the plot
    axes.set_xticks([])
    axes.set_yticks([])

    if cls_true!=None and cls_pred!=None:
      true = letters[cls_true]
      pred = letters[cls_pred]
      xlabel = 'True: {0}, Pred: {1}'.format(true,pred)
    elif cls_true!=None:
      true = letters[cls_true]
      xlabel = 'True: {0}'.format(true)
    elif cls_pred!=None:
      pred = letters[cls_pred]
      xlabel = 'Pred: {0}'.format(pred)
    else:
        xlabel = ''

    axes.set_xlabel(xlabel)
  else:
    for i,im in enumerate(ims):
      axes.flat[i].imshow(im.reshape(im.shape[0],im.shape[1]),cmap='binary')
      
      # Remove ticks from the plot
      axes.flat[i].set_xticks([])
      axes.flat[i].set_yticks([])
      
      if isinstance(cls_true,np.ndarray) and isinstance(cls_pred,np.ndarray):
        true = letters[cls_true[i][0]]
        pred = letters[cls_pred[i][0]]
        xlabel = 'True: {0}, Pred: {1}'.format(true,pred)
      elif isinstance(cls_true,np.ndarray):
        true = letters[cls_true[i][0]]
        xlabel = 'True: {0}'.format(true)
      elif isinstance(cls_pred,np.ndarray):
        pred = letters[cls_pred[i][0]]
        xlabel = 'Pred: {0}'.format(pred)
      else:
        xlabel = ''
      
      axes.flat[i].set_xlabel(xlabel)
  plt.show()

def plot_random(pred,num_label=False):
  pred = pred.reshape(-1,1)
  ind = np.arange(test.cls.shape[0])
  np.random.shuffle(ind)
  ind = ind[0:9]
  if num_label:
    plot_ims(ims=test.images[ind],cls_true=test.cls[ind],cls_pred=pred[ind])
  else:
    plot_ims_let(ims=test.images[ind],cls_true=test.cls[ind],cls_pred=pred[ind])

def plot_error(pred,inind=0,outind=9,num_label=False):
  pred = pred.reshape(-1,1)
  incorrect = (pred!=test.cls).reshape(-1)
  ind = np.arange(test.cls.shape[0])
  ind = ind[incorrect]
  ind = ind[inind:outind]
  if num_label:
    plot_ims(ims=test.images[ind],cls_true=test.cls[ind],cls_pred=pred[ind])
  else:
    plot_ims_let(ims=test.images[ind],cls_true=test.cls[ind],cls_pred=pred[ind])
    

def pil2np(img):
  """
  Convert RGB PIL images into a numpy array
  """
  out = np.fromstring(img.tobytes(),dtype=np.uint8)
  out = out.reshape((img.size[1],img.size[0],4))
  return(out)

def weights(shape,verb=False,name='weights'):
  """
  shape: [ker_h,ker_w,im_chan,num_ker]
  verb: Displays the info about the weights tensor
  """
  #w = tf.Variable(tf.truncated_normal(shape, stddev=0.05),name=name)
  # https://arxiv.org/pdf/1502.01852.pdf
  w = tf.Variable(tf.truncated_normal(shape, stddev=np.sqrt(2.0/(
                  shape[0]*shape[1]*shape[2]))),name=name)
  if verb:
    print(w)
  return(w)

def biases(shape,verb=False,name='biases'):
  """
  shape: [num_ker]
  verb: Displays the info about the biases tensor
  """
  b = tf.Variable(tf.constant(0.05,shape=shape),name=name)
  if verb:
    print(b)
  return(b)

def conv(inp,shape,strides=[1,1,1,1],padding='SAME',
           relu=True,dropout=False,do_prob=0.5,pool=False,name='conv',
           verb=False,histogram=True):
  """
  Tensorflow 2d convolution.
  
  inp: 4-D input tensor of shape [num,im_h,im_w,im_c]
  shape: Shape of the kernels. [ker_h,ker_w,inp_ch,num_k]
  strides: Strides of the convolution
  padding: 'SAME' - Adds zero-padding so the resulting convolution
                    is of the same size as the input
           'VALID' - No zero-padding, so it olny performs as many 
                     convolutions as it's possible, resulting in 
                     a different size.
  relu: Turn on relu as activation function. Else, it will not use any
  dropout: Turn on dropout
  do_prob: Select the percentage of the net to be turn off. [0.0 - 1.0]
  pool: Adds max pooling to the operation
  """
  with tf.name_scope(name) as scope:
    w = weights(shape,verb=verb)
    b = biases([shape[3]],verb=verb)

    conv = tf.nn.conv2d(input=inp,
                         filter=w,
                         strides=strides,
                         padding=padding,
                         name=name)

    conv += b

    if pool:
      conv = tf.nn.max_pool(value=conv,
                           ksize=[1,2,2,1],
                           strides=[1,2,2,1],
                           padding='SAME')
    if relu:
      conv = tf.nn.relu(conv)
      if histogram: 
        tf.summary.histogram('activations',conv)
    
    if dropout:
      conv = tf.nn.dropout(conv,do_prob)

    if histogram:
      tf.summary.histogram('weights',w)
      tf.summary.histogram('biases',b)

    return(conv)

def conv2(inp,shape,padding='SAME',strides=[1,1,1,1],relu=True,
          dropout=False,drop_prob=0.25,verb=False,histogram=False,
          name='conv'):
  """
  Tensorflow 2d convolution.
  
  inp: 4-D input tensor of shape [num,im_h,im_w,im_c]
  shape: Shape of the kernels. [ker_h,ker_w,inp_ch,num_k]
  strides: Strides of the convolution
  padding: 'SAME' - Adds zero-padding so the resulting convolution
                    is of the same size as the input
           'VALID' - No zero-padding, so it olny performs as many 
                     convolutions as it's possible, resulting in 
                     a different size.
  relu: Turn on relu as activation function. Else, it will not use any
  dropout: Turn on dropout
  do_prob: Select the percentage of the net to be turn off. [0.0 - 1.0]
  verb: Display information about the tensor.
  histogram: Indicates if information for tensorboard should be annexed.
  """
  with tf.name_scope(name) as scope:
    w = weights(shape,verb=verb)
    b = biases([shape[3]],verb=verb)

    conv = tf.nn.conv2d(input=inp,
                        filter=w,
                        strides=strides,
                        padding=padding,
                        name=name)
    out = conv + b

    if dropout:
      out = tf.nn.dropout(out,do_prob)
    
    if relu:
      out = tf.nn.relu(out)

    if verb:
      print(out)

    if histogram:
      tf.summary.histogram('activation',out)
      tf.summary.histogram('weights',w)
      tf.summary.histogram('biases',b)

    return(out)

def max_pool(layer,name='max_pool',ksize=[1,2,2,1],strides=[1,2,2,1],
  padding='SAME',args=False):
  with tf.name_scope(name) as scope:
    """
    layer: Tensor to be max pooled
    name: Name of the operation
    ksize: 1-D tensor with 4 elements. For more info, read below.
    strides 1-D tensor with 4 elements. For more info, read below.
    padding: 'SAME' - Adds zero padding.
             'VALID' - Does not add zero padding.
    args: Indicates if the indices for the maximum elements are required.


    Note on max_pool with/without args: If the max pool operation is 
      performed without indices/max args the output is a single tensor.
      However, if indices/max args are required, the output will be two
      tensors. The first one contains the resulting tensor of max pool,
      and the second one contains the indices for the max arguments.
    Note on ksize: Shape of [k on num_b,k on im_h,k on im_w,k on im_c].
      We could have a batch of RGB images with shape [6,68,68,3]
      and if ksize=[1,2,2,1] it will perform a max pooling over each 
      example. In each example it will perform a 2-by-2-window max 
      pooling over each of the three channels.
    Note on strides: Since we normally want to reduce the dimensionality
      of the tensor, we perform the max pooling on the image with its
      windows moving two possitions in the cols and rows, but not jumping
      any example nor any channel, due to strides=[1,2,2,1]
    """
    if not args:
      out = tf.nn.max_pool(value=layer,
                           ksize=ksize,
                           strides=[1,2,2,1],
                           padding=padding)
    else:
      out = tf.nn.max_pool_with_argmax(input=layer,
                                       ksize=ksize,
                                       strides=strides,
                                       padding=padding)
    return(out)

def flatten(layer):
  """
  Flattens a convolved tensor.... for the fully connected network
  """
  shape = layer.get_shape()

  num_features = shape[1:4].num_elements()

  layer_flat = tf.reshape(layer,[-1,num_features])

  return(layer_flat)

def fc(inp,shape,relu=True,logits=False,dropout=False,do_prob=0.5,
  verb=False,name='fc',histogram=True):
  """
  inp: input
  shape: [num_dim_in,num_class_out]
  """
  with tf.name_scope(name) as scope:
    w = weights(shape,verb=verb)
    b = biases([shape[1]],verb=verb)

    fc = tf.matmul(inp,w)
    fc += b

    if relu:
      fc = tf.nn.relu(fc)
      if histogram: 
        tf.summary.histogram('activations',conv)
    elif logits!=True:
      fc = tf.nn.softmax(fc)
    if dropout:
      fc = tf.nn.dropout(fc,do_prob)

    if histogram:
      tf.summary.histogram('weights',w)
      tf.summary.histogram('biases',b)

    return(fc)

def deconv(inp,out_like,shape,strides=[1,1,1,1],
  padding='SAME',relu=True,verb=False,name='deconv',dropout=False,
  do_prob=0.5,histogram=True):
  """
  inp: input tensor
  out_like: output-like shape tensor. What are the output tensor
      dimensions according to an existing tensor
  strides: strides used un the out_like in the convolution
  shape: [ker_h,ker_w,out_c,in_c]
  verb: Print weights and biases shape
  """
  with tf.name_scope(name) as scope:
    w = weights(shape,verb=verb)
    b = biases([shape[2]],verb=verb)
    out_shape = tf.shape(out_like)
    
    transpose_conv = tf.nn.conv2d_transpose(value=inp,
                                           filter=w,
                                           output_shape=out_shape,
                                           strides=strides,
                                           padding=padding)
    
    transpose_conv += b
    #print(transpose_conv.get_shape())
    if relu:
      transpose_conv = tf.nn.relu(transpose_conv)
      if histogram:
        tf.summary.histogram('activations',transpose_conv)
    if dropout:
      transpose_conv = tf.nn.dropout(transpose_conv,do_prob)

    if histogram:
      tf.summary.histogram('weights',w)
      tf.summary.histogram('biases',b)
    
    return(transpose_conv)

def deconv2(inp,shape,strides=[1,1,1,1],padding='SAME',relu=False,
  verb=False,name='deconv',dropout=False,do_prob=0.5,histogram=True):
  """
  """
  with tf.name_scope(name) as scope:
    w = weights(shape,verb=verb)
    b = biases([shape[2]],verb=verb)

    x_shape = tf.shape(inp)
    out_shape = tf.stack([x_shape[0],x_shape[1],x_shape[2],shape[2]])

    transpose_conv = tf.nn.conv2d_transpose(value=inp,
                                            filter=w,
                                            output_shape=out_shape,
                                            strides=strides,
                                            padding=padding)

    transpose_conv += b

    if relu:
      transpose_conv = tf.nn.relu(transpose_conv)
      if histogram:
        tf.summary.histogram('activations',transpose_conv)
    if dropout:
      transpose_conv = tf.nn.dropout(transpose_conv,do_prob)

    if histogram:
      tf.summary.histogram('weights',w)
      tf.summary.histogram('biases',b)

    return(transpose_conv)


###### STARTS
# https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation/blob/master/DeconvNet.py
def unravel_argmax(argmax, shape):
  output_list = []
  output_list.append(argmax // (shape[2] * shape[3]))
  output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
  return(tf.stack(output_list))

def unpool_layer2x2(x, raveled_argmax, out_shape,verb=False,name='unpool'):
  with tf.name_scope(name) as scope:
    argmax = unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
    output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])
    if verb:
      print(output)

    height = tf.shape(output)[0]
    if verb:
      print(height)
    width = tf.shape(output)[1]
    if verb:
      print(width)
    channels = tf.shape(output)[2]
    if verb:
      print(channels)

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])
    if verb:
      print(t1)

    t2 = tf.squeeze(argmax)
    t2 = tf.stack((t2[0], t2[1]), axis=0)
    t2 = tf.transpose(t2, perm=[3, 1, 2, 0])
    if verb:
      print(t2)

    t = tf.concat([t2, t1], 3)
    if verb:
      print(t)
    indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])
    if verb:
      print(indices)

    x1 = tf.squeeze(x)
    x1 = tf.reshape(x1, [-1, channels])
    x1 = tf.transpose(x1, perm=[1, 0])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
    return(tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0))

##### END

def unpool_with_with_argmax(pooled,ind,input_shape,ksize=[1, 2, 2, 1],
                  name='unpool'):
  """
  https://github.com/sangeet259/tensorflow_unpooling
    To unpool the tensor after  max_pool_with_argmax.
    Argumnets:
        pooled:    the max pooled output tensor
        ind:       argmax indices , the second output of max_pool_with_argmax
        ksize:     ksize should be the same as what you have used to pool
    Returns:
        unpooled:      the tensor after unpooling
    Some points to keep in mind ::
        1. In tensorflow the indices in argmax are flattened, so that a maximum value at position [b, y, x, c] 
           becomes flattened index ((b * height + y) * width + x) * channels + c
        2. Due to point 1, use broadcasting to appropriately place the values at their right locations ! 
  """
  with tf.name_scope(name) as scope:
    # Get the the shape of the tensor in th form of a list
    #input_shape = pooled.get_shape().as_list()

    # Determine the output shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # Ceshape into one giant tensor for better workability
    pooled_ = tf.reshape(pooled, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]])
    # The indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes flattened index ((b * height + y) * width + x) * channels + c
    # Create a single unit extended cuboid of length bath_size populating it with continous natural number from zero to batch_size
    tmp_shape = np.array([input_shape[0], 1, 1, 1],dtype=np.int64)
    batch_range = tf.reshape(tf.range(tf.cast(output_shape[0],tf.int64), dtype=ind.dtype), shape=tmp_shape)  
    b = tf.ones_like(ind) * batch_range
    b_ = tf.reshape(b, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
    ind_ = tf.reshape(ind, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
    ind_ = tf.concat([b_, ind_],1)
    ref = tf.Variable(tf.zeros([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]))
    # Update the sparse matrix with the pooled values , it is a batch wise operation
    unpooled_ = tf.scatter_nd_update(ref, ind_, pooled_)
    # Reshape the vector to get the final result 
    unpooled = tf.reshape(unpooled_, [output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
    return(unpooled)


def unpool_with_argmax(pooled,ind,input_shape, ksize=[1, 2, 2, 1],
            name='unpool'):
  """
    ORIGINAL
    To unpool the tensor after  max_pool_with_argmax.
    Argumnets:
        pooled:    the max pooled output tensor
        ind:       argmax indices , the second output of max_pool_with_argmax
        ksize:     ksize should be the same as what you have used to pool
    Returns:
        unpooled:      the tensor after unpooling
    Some points to keep in mind ::
        1. In tensorflow the indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes flattened index ((b * height + y) * width + x) * channels + c
        2. Due to point 1, use broadcasting to appropriately place the values at their right locations ! 
  """
  with tf.name_scope(name) as scope:
    # Get the the shape of the tensor in th form of a list
    #input_shape = pooled.get_shape().as_list()
    #print(input_shape)
    # Determine the output shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # Ceshape into one giant tensor for better workability
    #pooled_ = tf.reshape(pooled, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]])
    pooled_ = tf.reshape(pooled, [-1])
    # The indices in argmax are flattened, so that a maximum value at position [b, y, x, c] becomes 
    # flattened index ((b * height + y) * width + x) * channels + c
    # Create a single unit extended cuboid of length bath_size populating it with continous natural number from zero to batch_size

    #tf.placeholder(tf.int64,shape=[None,])
    #batch_range = tf.reshape(tf.range(tf.shape(output_shape,out_type=tf.int64)[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
    tmp_out1 = tf.placeholder(tf.int64,shape=[output_shape[0]])
    print('HERE {0}'.format(tmp_out1))
    batch_range = tf.reshape(tf.range(tf.shape(tmp_out1,out_type=tf.int64)[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b_ = tf.reshape(b, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
    ind_ = tf.reshape(ind, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
    ind_ = tf.concat([b_, ind_],1)
    
    tmp_ref = tf.placeholder(tf.float32,shape=[output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])
    #ref = tf.Variable(tf.zeros([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]))
    ref = tf.Variable(tf.zeros_like(tmp_ref))
    # Update the sparse matrix with the pooled values , it is a batch wise operation
    unpooled_ = tf.scatter_nd_update(ref, ind_, pooled_)
    # Reshape the vector to get the final result 
    unpooled = tf.reshape(unpooled_, [output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
    return(unpooled)