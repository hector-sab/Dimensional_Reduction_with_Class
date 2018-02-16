import os
import numpy as np
# Hidde debug infro from tf... https://stackoverflow.com/a/38645250/5969548
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

def weights(shape,verb=False,name='weights'):
  """
  shape: [ker_h,ker_w,im_chan,num_ker]
  verb: Displays the info about the weights tensor
  """
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

def conv(inp,shape,padding='SAME',strides=[1,1,1,1],relu=False,
  dropout=False,drop_prob=0.8,verb=False,histogram=False,
  name='conv',l2=False):
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

    if l2:
      l2_reg = tf.nn.l2_loss(w)
      return(out,l2_reg)
    else:
      return(out)

def dilated_conv(inp,shape,rate=2,relu=False,histogram=False,
	l2=False,padding='SAME',name='dilated_conv'):
  """
  shape: Shape of the kernels. [ker_h,ker_w,inp_ch,num_k]
  """
  with tf.name_scope(name) as scope:
    w = weights(shape)
    b = biases([shape[3]])
    
    atrous = tf.nn.atrous_conv2d(value=inp,
                                 filters=w,
                                 rate=rate,
                                 padding=padding,
                                 name=name)
    out = atrous + b
    
    if relu:
      out = tf.nn.relu(out)
    
      
    if histogram:
      tf.summary.histogram('activation',out)
      tf.summary.histogram('weights',w)
      tf.summary.histogram('biases',b)

    if l2:
      l2_reg = tf.nn.l2_loss(w)
      return(out,l2_reg)
    else:
    	return(out)

def dilated_deconv(inp,shape,rate=2,strides=[1,1,1,1],relu=False,l2=False,
	verb=False,padding='SAME',name='dilated_deconv'):
	"""
	TODO: Check if strides 2,2 works
	"""
	with tf.name_scope(name) as scope:
		w = weights(shape,verb=verb)
		b = biases([shape[2]],verb=verb)

		x_shape = tf.shape(inp)
		out_shape = tf.stack([x_shape[0],x_shape[1]*strides[1],
                          x_shape[2]*strides[2],shape[2]])

		out = tf.nn.atrous_conv2d_transpose(value=inp,
																			filter=w,
																			output_shape=out_shape,
																			rate=rate,
																			padding='SAME',
																			name=name)
		out += b

		#out = tf.reshape(out,shape=[-1,inp.get_shape()[1].value*strides[1],
    #  inp.get_shape()[2].value*strides[2],shape[2]])

		if relu:
      out = tf.nn.relu(out)
      if histogram:
        tf.summary.histogram('activations',out)

    if histogram:
      tf.summary.histogram('weights',w)
      tf.summary.histogram('biases',b)

    if l2:
      l2_reg = tf.nn.l2_loss(w)
      return(transpose_conv,l2_reg)
    else:
      return(transpose_conv)




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

def fc(inp,shape,relu=False,logits=False,dropout=False,do_prob=0.5,
  verb=False,name='fc',histogram=True,l2=False):
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

    if l2:
      l2_reg = tf.nn.l2_loss(w)
      return(fc,l2_reg)
    else:
      return(fc)

def deconv(inp,shape,strides=[1,1,1,1],padding='SAME',relu=False,
  verb=False,name='deconv',dropout=False,drop_prob=0.8,histogram=True,
  l2=False):
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

    x_shape = tf.shape(inp)
    out_shape = tf.stack([x_shape[0],x_shape[1]*strides[1],
                          x_shape[2]*strides[2],shape[2]])

    transpose_conv = tf.nn.conv2d_transpose(value=inp,
                                            filter=w,
                                            output_shape=out_shape,
                                            strides=strides,
                                            padding=padding)

    transpose_conv += b

    # Just so it conservs shape information
    transpose_conv = tf.reshape(transpose_conv,shape=[-1,inp.get_shape()[1].value*strides[1],
      inp.get_shape()[2].value*strides[2],shape[2]])

    if relu:
      transpose_conv = tf.nn.relu(transpose_conv)
      if histogram:
        tf.summary.histogram('activations',transpose_conv)
    if dropout:
      transpose_conv = tf.nn.dropout(transpose_conv,drop_prob)

    if histogram:
      tf.summary.histogram('weights',w)
      tf.summary.histogram('biases',b)



    if l2:
      l2_reg = tf.nn.l2_loss(w)
      return(transpose_conv,l2_reg)
    else:
      return(transpose_conv)

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
    tmp_range = tf.range(tf.shape(tmp_out1,out_type=tf.int64)[0],dtype=ind.dtype)
    batch_range = tf.reshape(tmp_range, shape=[-1, 1, 1, 1])
    



    b = tf.ones_like(ind) * batch_range
    b_ = tf.reshape(b, [-1, 1])
    ind_ = tf.reshape(ind, [-1, 1])
    ind_ = tf.concat([b_, ind_],1)
    
    tmp_ref = tf.placeholder(tf.float32,shape=[output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])
    #ref = tf.Variable(tf.zeros([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]))
    #tmp_zeros = tf.zeros_like(tmp_ref)
    ref = tf.zeros_like(tmp_ref)
    ref = tf.Variable(ref)
    # Update the sparse matrix with the pooled values , it is a batch wise operation
    unpooled_ = tf.scatter_nd_update(ref, ind_, pooled_)
    # Reshape the vector to get the final result 
    unpooled = tf.reshape(unpooled_, [output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
    return(unpooled)