# coding=utf-8
"""
Author: Héctor Sánchez
Date: January-31-2018
Description: Contains the different models for MNIST segmentation
"""
import os
import sys
# Hidde debug infro from tf... https://stackoverflow.com/a/38645250/5969548
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import utils as ut


class SegModel:
  """
  This object will select one model and train it for the purpose of 
  semantic segmentation
  """
  # TODO: Check if visualization nodes can be reused to create just one
  def __init__(self,train,val,test=None,num_class=11,ex=None,model=0,
    bs=1,lr=3e-5,dropout=False,drop_prob=0.8,training=True,save=False,
    save_dir=None,save_checkp=None,max_to_keep=1,load=False,load_dir=None,
    load_checkp=None,save_load_same=True,load_step=None,tb_log=False,
    log_dir=None,log_name=None,l2=True,version=1):
    """
    -train: Triaining data using the class DataSeg
    -val: Validation data using the class DataSeg
    -test: Test data using the class DataSeg
    -num_class: Number of classes. I.E. MNIST has 10 classes
    -ex: Number of examples in the input tensor of the network
    -model: Model used to train/predict
    -bs: Number of example per batch at training time
    -lr: Learning rate
    -dropout: Indicates if dropout will be used
    -drop_prob: Percentaje of neurons to be set to zero
    -training: Indicates if the model will be trained
    -save: Flag indicating if the model will save checkpoints
    -save_dir: Directory where the model will be saved
    -save_checkp: Name of the checkpoint to be saved
    -max_to_keep: How many checkpoints should be saved
    -load: Flag indicating if the model will load checkpoints
    -load_dir: Directory where the pre-trained model is located
    -load_checkp: Name of the checkpoint to be loaded
    -save_load_same: Flag indicating if the load and save model 
              are the same
    -load_step: if the model as many checkpoints steps, select which
              one should be loaded
    -tb_log: Flag indicating if summaries will be saved
    -log_dir: Directory where it will be saved
    -log_name: Name of the summary
    -l2: Activates l2 regularizatoin
    -version: Indicates  which version of a same net we are executing.
        It just affects the nameming of the directories where data is
        stored
    """
    self.session = tf.Session()
    # Data base
    self.train = train
    self.val = val
    self.test = test

    # Parameters
    self.ex = ex 
    self.bs = bs # Number of examples per batch
    self.lr = lr
    self.dropout = dropout
    self.drop_prob = drop_prob
    self.tb_log = tb_log
    self.training = training
    self.save = save
    self.load = load
    self.load_step = load_step

    self.total_it = 0
    self.best_acc = 0

    self.l2 = l2

    # Image specs
    self.im_h = self.train.images.shape[1]
    self.im_w = self.train.images.shape[2]
    self.im_c = self.train.images.shape[3]

    self.num_class = num_class

    # Create input placeholders
    self.inputs()

    # Select the model to train
    if model==0:
      self.model = ModelStv1(inp=self.x,dropout=self.dropout,
        drop_prob=self.drop_prob,histogram=self.tb_log,
        num_class=self.num_class,version=version)
    elif model==1:
      self.model = ModelMPv1(inp=self.x,dropout=self.dropout,
        drop_prob=self.drop_prob,histogram=self.tb_log,
        num_class=self.num_class,version=version)
    else:
      print("There's no model with that option choice...")
      sys.exit()

    # Specify where to save the model and the tb log
    save_dir,save_checkp = self.model.checkpoint_dir(save_dir,save_checkp)
    load_dir,load_checkp = self.model.checkpoint_dir(load_dir,load_checkp)
    log_dir,log_name = self.model.log_dir(log_dir,log_name)

    self.last_layer = self.model.last_layer()

    # Create output placeholders
    self.outputs()

    if self.training:
      self.trainable()

    self.summary = tf.summary.merge_all()

    if self.save or self.load:
      self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    self.init_variables()

    """ # I belive it must always be executed... let's see...
    if self.save:
      self.savable(save_dir,save_checkp)
    """
    self.savable(save_dir,save_checkp)
    if self.load:
      self.loadable(load_dir,load_checkp,save_load_same)

    if self.tb_log:
      self.tensorboard_log(log_dir,log_name)


  def optimize(self,num_it=0,verb=None,tb_log_it=100):
    """
    Description: Trains the model

    num_it: Number of iterations to train
    verb: Display training process each 'verb' iterations
    tb_log_it: Saves summary each 'tb_log_it' if log is True
    """
    print('\nStarting optimization...\n')
    for it in range(num_it):
      self.total_it += 1

      data = self.train.next_batch(self.bs)
      if self.dropout:
        feed_dict = {self.x: data['ims'], 
                     self.y_seg: data['seg'],
                     self.drop_prob: self.drop_prob}
      else:
        feed_dict = {self.x: data['ims'], 
                     self.y_seg: data['seg']}

      self.session.run(self.optimizer,feed_dict=feed_dict)

      if verb is not None and self.total_it%verb==0:
        acc = self.full_acc(self.val,bs=self.bs)

        if self.best_acc<acc:
          self.best_val_acc = acc
          saved_str = '*'

          if self.save:
            self.saver.save(sess=self.session,save_path=self.save_path,
              global_step=self.total_it)
        else:
          saved_str = ''
        msg = 'It: {0}/{1} - Acc {2:.2%} {3}'.format(self.total_it,
                self.total_it+num_it-(it+1),acc,saved_str)
        print(msg)

      if verb is None and self.save:
        """
        In case verbose is disabled, to ensure checkpoints are saved
        """
        if self.total_it%100==0:
          acc = self.full_acc(self.val,self.bs)
          if self.best_acc<acc:
            self.best_val_acc = acc
            self.saver.save(sess=self.session,save_path=self.save_path,
                global_step=self.total_it)

      if self.tb_log and self.total_it%tb_log_it==0:
        self.val.restart_next_batch()
        data = self.val.next_batch(self.bs)
        tmp_feed = {self.x:data['ims'], self.y_seg:data['seg']}
        s = self.session.run(self.summary,feed_dict=tmp_feed)
        self.writer.add_summary(s,self.total_it)

  def test_acc(self,bs=1):
    """
    Description: Returns the accuracy on test set
    """
    acc = self.full_acc(self.test,bs)
    return(acc)

  def full_acc(self,dataset,bs=1):
    """
    Description: Returns the validation accuracy when batch size
            is equal to bs (one...).

    data: DataSet used to calculate full accuracy. It can be
          train, val, or test
    """
    dataset.restart_next_batch()
    num_ex = dataset.images.shape[0]
    total_acc = 0
    
    for it in range(int(num_ex/bs)):
      data = dataset.next_batch(bs)
      if self.ex is not None and data['ims'].shape[0]<self.ex:
        # Ensures that won't be an error caused by shape incompatibility
        continue
      feed_dict = {self.x: data['ims'],
                   self.y_seg: data['seg']}
      acc = self.session.run(self.accuracy,feed_dict=feed_dict)
      total_acc += acc
    total_acc /= num_ex

    return(total_acc)

  def tensorboard_log(self,log_dir,log_name):
    """
    Description: Used to creat a log  in tensorboard
    """
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)
    self.writer = tf.summary.FileWriter(log_dir+log_name)
    msg = '\nSaving Tensorboard log at: {0}{1}'.format(log_dir,log_name)
    print(msg)
    self.writer.add_graph(self.session.graph)

  def savable(self,save_dir,save_checkp):
    """
    Save path to be saved
    """
    if self.save:
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      msg = '\nSaving checkpoints at: {0}{1}'.format(save_dir,save_checkp)
      print(msg)
    self.save_path = os.path.join(save_dir,save_checkp)

  def loadable(self,load_dir,load_checkp,save_load_same):
    """
    Load path to be saved
    """
    if save_load_same:
      load_path = self.save_path
    else:
      load_path = os.path.join(load_dir,load_checkp)

    msg = '\nLoading checkpoint: {0}'.format(load_path)
    print(msg)
    self.restore_variables(load_path,self.load_step)

  def init_variables(self):
    """
    Initialize all weights in the graph
    """
    self.session.run(tf.global_variables_initializer())
  
  def restore_variables(self,load_path,load_step=None):
    """
    Restores all weights values from previous training
    """
    if self.load_step is None:
      self.saver.restore(sess=self.session, save_path=load_path)
    else:
      self.saver.restore(sess=self.session, save_path=load_path+'-'+str(load_step))
      self.total_it = load_step

  def trainable(self):
    """
    Creates tensors needed for training the model
    """

    # TODO: beta value accessible from somewhere else
    beta = 0.01
    with tf.name_scope('cross_entropy'):
      self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=self.logits,labels=self.y_seg_onehot,name='cross_entropy')
      self.cost = tf.reduce_mean(self.cross_entropy,name='cost')
      if self.l2:
        for i,reg in enumerate(self.model.reg):
          self.cost += tf.reduce_mean(beta*reg,name='cost-w'+str(i))
      tf.summary.scalar('cost',self.cost)

    with tf.name_scope('train'):
      self.optimizer = tf.train.AdamOptimizer(
        learning_rate=self.lr).minimize(self.cost)

      # TODO: Fix batch norm
      ### STARTS: For batch norm... mean and variance
      #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      #
      #with tf.control_dependencies(update_ops):
      #  train_ops = [self.optimizer] + update_ops
      #  train_op_final = tf.group(*train_ops)
      ### ENDS: For batch norm... mean and variance

    with tf.name_scope('accuracy'):
      self.correct_prediction = tf.equal(self.y_pred_cls,self.y_seg_cls,
                                    name='correct_prediction')
      self.accuracy = tf.reduce_mean(
        tf.cast(self.correct_prediction, tf.float32))
      tf.summary.scalar('accuracy',self.accuracy)
  
  def outputs(self):
    """
    Creates all placeholders needed to predict segmentation
    """
    with tf.name_scope('Prediction'):
      self.logits = tf.reshape(self.last_layer,
        shape=[-1,self.num_class],name='logits_rs')
      self.y_pred = tf.nn.softmax(self.logits,name='y_pred')
      self.y_pred_cls = tf.argmax(self.y_pred,axis=1)
      self.y_pred_cls = tf.reshape(self.y_pred_cls,shape=[-1,1],
        name='y_pred_cls')
      self.y_pred_cls_seg = tf.reshape(self.y_pred_cls,
        shape=[-1,self.im_h,self.im_w,1],name='y_pred_cls_seg')
      
      ### START: Seg Image visualization
      self.seg_out_vis = tf.divide(self.y_pred_cls_seg,
        self.num_class)
      self.seg_out_vis = tf.cast(tf.scalar_mul(255,self.seg_out_vis),
        tf.uint8)
      tf.summary.image('seg_output',self.seg_out_vis,1)
      ### END: Seg Image visualization

      msg = '\n\t{0} \n\t{1} \n\t{2} \n\t{3}'
      msg = msg.format(self.logits,self.y_pred,
              self.y_pred_cls,self.y_pred_cls_seg)
      print(msg)

  def inputs(self):
    """
    Initialize all required input placeholders
    """
    with tf.name_scope('Input'):
      self.drop_prob = tf.placeholder_with_default(1.0,shape=[],
        name='drop_prob')
      self.x = tf.placeholder(tf.float32,\
        shape=[self.ex,self.im_h,self.im_w,self.im_c],name='x')
      tf.summary.image('input',self.x,1)

      # TODO: Fix batch norm
      # Batch normalization
      #self.x = tf.layers.batch_normalization(inputs=self.x_)

      # Image with pixel-lebel labels
      self.y_seg = tf.placeholder(tf.int64,\
        shape=[self.ex,self.im_h,self.im_w,1],name='y_seg')
      self.y_seg_cls = tf.reshape(self.y_seg,shape=[-1,1],\
        name='y_seg_cls')

      # Image with 'num_seg_class' channels
      self.y_seg_onehot = tf.one_hot(self.y_seg_cls,\
        depth=self.num_class,axis=1)
      # Reshape to errase a useless extra dimension
      self.y_seg_onehot = tf.reshape(self.y_seg_onehot,\
        shape=[-1,self.num_class],name='y_seg_onehot')

      ### START: Seg Image visualization
      self.seg_inp_vis = tf.divide(self.y_seg,self.num_class)
      self.seg_inp_vis = tf.cast(tf.scalar_mul(255,self.seg_inp_vis),
        tf.uint8)
      tf.summary.image('seg input',self.seg_inp_vis,1)
      ### END: Seg Image visualization

      msg = '\n\t{0} \n\t{1} \n\t{2} \n\t{3}'
      msg = msg.format(self.x,self.y_seg,self.y_seg_cls,self.y_seg_onehot)
      print(msg)


class ModelMPv1:
  """
  Contains the model of the segmentation using max pooling
  """
  def __init__(self,inp,dropout=False,drop_prob=0.25,
    histogram=True,num_class=11,def_cp_name='mnist_seg',
    def_cp_path='checkpoints/mnist_seg_mp',
    def_log_name='mnist_seg_mp',
    def_log_path='./log/',version=1):
    """
    inp: Input placeholder.
    shape: Tensorflow tensor shape used in the input placeholder.
           It must be a list object.
    dropout: Flag used to indicate if dropout will be used
    drop_prob: Percentage of neurons to be turned off
    histogram: Indicates if information for tensorboard should be annexed.
    """
    self.def_cp_name = def_cp_name
    self.def_cp_path = def_cp_path+'v'+str(version)
    if self.def_cp_path[-1]!='/':
      self.def_cp_path += '/'
    self.def_log_name = def_log_name+'v'+str(version)
    self.def_log_path = def_log_path
    if self.def_log_path[-1]!='/':
      self.def_log_path += '/'
    self.reg = [] # Contains l2 regularizaton for weights
    self.dropout = dropout
    self.drop_prob = drop_prob
    self.x = inp
    # shape vs get_shape https://stackoverflow.com/a/43290897/5969548
    self.im_h = int(self.x.get_shape()[1])
    self.im_w = int(self.x.get_shape()[2])
    self.im_c = int(self.x.get_shape()[3])
    self.num_class = num_class
    ##### Network Specs
    ks1 = 3
    num_k1 = 8
    ks2 = 3
    num_k2 = 8

    ks3 = 3
    num_k3 = 16
    ks4 = 3
    num_k4 = 16

    ks5 = 3
    num_k5 = 32
    ks6 = 3
    num_k6 = 32

    ##### Core model
    c1_shape = [ks1,ks1,self.im_c,num_k1]
    self.conv1,reg = ut.conv2(inp=self.x,shape=c1_shape,name='conv1',
      dropout=self.dropout,drop_prob=self.drop_prob,histogram=histogram,
      l2=True)
    self.reg.append(reg)
    
    c2_shape = [ks2,ks2,num_k1,num_k2]
    self.conv2,reg = ut.conv2(inp=self.conv1,shape=c2_shape,
      name='conv2',dropout=self.dropout,drop_prob=self.drop_prob,
      histogram=histogram,l2=True)
    self.reg.append(reg)

    self.pool1,self.ind1 = ut.max_pool(self.conv2,args=True,
      name='maxpool1')

    c3_shape = [ks3,ks3,num_k2,num_k3]
    self.conv3,reg = ut.conv2(inp=self.pool1,shape=c3_shape,name='conv3',
      dropout=self.dropout,drop_prob=self.drop_prob,histogram=histogram,
      l2=True)
    self.reg.append(reg)
    
    c4_shape = [ks4,ks4,num_k3,num_k4]
    self.conv4,reg = ut.conv2(inp=self.conv3,shape=c4_shape,name='conv4',
      dropout=self.dropout,drop_prob=self.drop_prob,histogram=histogram,
      l2=True)
    self.reg.append(reg)

    self.pool2,self.ind2 = ut.max_pool(self.conv4,args=True,
      name='maxpool2')

    c5_shape = [ks5,ks5,num_k4,num_k5]
    self.conv5,reg = ut.conv2(inp=self.pool2,shape=c5_shape,name='conv5',
      dropout=self.dropout,drop_prob=self.drop_prob,histogram=histogram,
      l2=True)
    self.reg.append(reg)

    c6_shape = [ks6,ks6,num_k5,num_k6]
    self.conv6,reg = ut.conv2(inp=self.conv5,shape=c6_shape,name='conv6',
      dropout=self.dropout,drop_prob=self.drop_prob,histogram=histogram,
      l2=True)
    self.reg.append(reg)



    d1_shape = [ks6,ks6,num_k6,num_k6]
    self.deconv1,reg = ut.deconv2(inp=self.conv6,shape=d1_shape,
      relu=True,name='deconv1',dropout=self.dropout,
      drop_prob=self.drop_prob,histogram=histogram,l2=True)
    self.reg.append(reg)

    d2_shape = [ks5,ks5,num_k4,num_k5]
    self.deconv2,reg = ut.deconv2(inp=self.deconv1,shape=d2_shape,
      relu=True,name='deconv2',dropout=self.dropout,
      drop_prob=self.drop_prob,histogram=histogram,l2=True)
    self.reg.append(reg)

    self.unpool1 = ut.unpool_with_argmax(self.deconv2,self.ind2,
                      input_shape=[1,7,7,num_k4],
                      name='unpool1')

    #self.sum1 = self.unpool1 + self.conv4
    d3_shape = [ks4,ks4,num_k3,num_k4]
    self.deconv3,reg = ut.deconv2(inp=self.unpool1,shape=d3_shape,
      relu=True,name='deconv3',dropout=self.dropout,
      drop_prob=self.drop_prob,histogram=histogram,l2=True)
    self.reg.append(reg)

    self.sum1 = self.deconv3 + self.conv4
    d4_shape = [ks3,ks3,num_k2,num_k3]
    self.deconv4,reg = ut.deconv2(inp=self.deconv3,shape=d4_shape,
      relu=True,name='deconv4',dropout=self.dropout,
      drop_prob=self.drop_prob,histogram=histogram,l2=True)
    self.reg.append(reg)

    self.unpool2 = ut.unpool_with_argmax(self.deconv4,self.ind1,
                      input_shape=[self.x.get_shape()[0].value,14,14,num_k2],
                      name='unpool2')

    #self.sum2 = self.unpool2 + self.conv2
    d5_shape = [ks2,ks2,num_k2,num_k2]
    self.deconv5,reg = ut.deconv2(inp=self.unpool2,shape=d5_shape,
      relu=True,name='deconv5',dropout=self.dropout,
      drop_prob=self.drop_prob,histogram=histogram,l2=True)
    self.reg.append(reg)

    self.sum2 = self.deconv5 + self.conv2
    d6_shape = [ks1,ks1,self.num_class,num_k2]
    self.deconv6,reg = ut.deconv2(inp=self.sum2,shape=d6_shape,
                   relu=False,name='deconv6',histogram=histogram,l2=True)
    self.reg.append(reg)


    self.pre_logits = self.deconv6

    msg = '\n\t{0} \n\t{1} \n\t{2} \n\t{3} \n\t{4} \n\t{5} '
    msg += '\n\t{6} \n\t{7}'
    msg = msg.format(self.conv1,self.conv2,self.pool1,self.conv3,
                     self.conv4,self.pool2,self.conv5,self.conv6)
    msg += '\n\t{0} \n\t{1} \n\t{2} \n\t{3} \n\t{4} \n\t{5} '
    msg += '\n\t{6} \n\t{7}'
    msg = msg.format(self.deconv1,self.deconv2,self.unpool1,
                     self.deconv3,self.deconv4,self.unpool2,
                     self.deconv5,self.deconv6)
    print(msg)

  def last_layer(self):
    """
    Returns the last layer of the model
    """
    return(self.pre_logits)

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



class ModelStv1:
  """
  Contains the model of the segmentation using strides of two,
  and no max pooling
  """
  def __init__(self,inp,dropout=False,drop_prob=0.25,
    histogram=True,num_class=11,verb=True,def_cp_name='mnist_seg',
    def_cp_path='checkpoints/mnist_seg_st',def_log_name='mnist_seg_st',
    def_log_path='./log/',version=1):
    """
    inp: Input placeholder.
    shape: Tensorflow tensor shape used in the input placeholder.
           It must be a list object.
    dropout: Flag used to indicate if dropout will be used
    drop_prob: Percentage of neurons to be turned off
    histogram: Indicates if information for tensorboard should be annexed.
    """
    self.def_cp_name = def_cp_name
    self.def_cp_path = def_cp_path+'v'+str(version)
    if self.def_cp_path[-1]!='/':
      self.def_cp_path += '/'
    self.def_log_name = def_log_name+'v'+str(version)
    self.def_log_path = def_log_path
    if self.def_log_path[-1]!='/':
      self.def_log_path += '/'
    self.reg = [] # Contains l2 regularizaton for weights
    self.dropout = dropout
    self.drop_prob = drop_prob
    self.x = inp
    # shape vs get_shape https://stackoverflow.com/a/43290897/5969548
    self.im_h = int(self.x.get_shape()[1])
    self.im_w = int(self.x.get_shape()[2])
    self.im_c = int(self.x.get_shape()[3])
    self.num_class = num_class
    
    ##### Network Specs
    ks1 = 3; num_k1 = 8
    ks2 = 3; num_k2 = 8

    ks3 = 3; num_k3 = 16
    ks4 = 3; num_k4 = 16

    ks5 = 3; num_k5 = 32
    ks6 = 3; num_k6 = 32

    #### Core Model
    c1_shape = [ks1,ks1,self.im_c,num_k1]
    self.conv1,reg = ut.conv2(inp=self.x,shape=c1_shape,name='conv1',
      dropout=self.dropout,drop_prob=self.drop_prob,histogram=histogram,
      l2=True)
    self.reg.append(reg)

    c2_shape = [ks2,ks2,num_k1,num_k2]
    self.conv2,reg = ut.conv2(inp=self.conv1,shape=c2_shape,
      name='conv2',dropout=self.dropout,
      drop_prob=self.drop_prob,histogram=histogram,l2=True)
    self.reg.append(reg)

    c3_shape = [ks3,ks3,num_k2,num_k3]
    self.conv3,reg = ut.conv2(inp=self.conv2,shape=c3_shape,name='conv3',
      strides=[1,2,2,1],dropout=self.dropout,drop_prob=self.drop_prob,histogram=histogram,
      l2=True)
    self.reg.append(reg)

    c4_shape = [ks4,ks4,num_k3,num_k4]
    self.conv4,reg = ut.conv2(inp=self.conv3,shape=c4_shape,
      name='conv4',dropout=self.dropout,
      drop_prob=self.drop_prob,histogram=histogram,l2=True)
    self.reg.append(reg)

    c5_shape = [ks5,ks5,num_k4,num_k5]
    self.conv5,reg = ut.conv2(inp=self.conv4,shape=c5_shape,name='conv5',
      strides=[1,2,2,1],dropout=self.dropout,drop_prob=self.drop_prob,histogram=histogram,
      l2=True)
    self.reg.append(reg)

    c6_shape = [ks6,ks6,num_k5,num_k6]
    self.conv6,reg = ut.conv2(inp=self.conv5,shape=c6_shape,name='conv6',
      dropout=self.dropout,drop_prob=self.drop_prob,histogram=histogram,
      l2=True)
    self.reg.append(reg)



    d1_shape = [ks6,ks6,num_k6,num_k6]
    self.deconv1,reg = ut.deconv2(inp=self.conv6,shape=d1_shape,
      relu=True,dropout=self.dropout,drop_prob=self.drop_prob,
      histogram=histogram,name='deconv1',l2=True)
    self.reg.append(reg)

    #d2_shape = [ks5,ks5,num_k4,num_k5]
    d2_shape = [ks5,ks5,num_k5,num_k6]
    self.deconv2,reg = ut.deconv2(inp=self.deconv1,shape=d2_shape,
      relu=True,name='deconv2',dropout=self.dropout,
      drop_prob=self.drop_prob,histogram=histogram,l2=True)
    self.reg.append(reg)

    #self.sum1 = self.deconv2 + self.conv4
    #d3_shape = [ks4,ks4,num_k3,num_k4]
    d3_shape = [ks4,ks4,num_k4,num_k5]
    self.deconv3,reg = ut.deconv2(inp=self.deconv2,shape=d3_shape,
      relu=True,strides=[1,2,2,1],name='deconv3',dropout=self.dropout,
      drop_prob=self.drop_prob,histogram=histogram,l2=True)
    self.reg.append(reg)

    self.sum1 = self.deconv3 + self.conv4
    #d4_shape = [ks3,ks3,num_k2,num_k3]
    d4_shape = [ks3,ks3,num_k3,num_k4]
    self.deconv4,reg = ut.deconv2(inp=self.sum1,shape=d4_shape,
      relu=True,name='deconv4',dropout=self.dropout,
      drop_prob=self.drop_prob,histogram=histogram,l2=True)
    self.reg.append(reg)

    #self.sum2 = self.deconv4 + self.conv3
    #d5_shape = [ks2,ks2,num_k2,num_k2]
    d5_shape = [ks2,ks2,num_k1,num_k3]
    self.deconv5,reg = ut.deconv2(inp=self.deconv4,shape=d5_shape,
      strides=[1,2,2,1],relu=True,name='deconv5',dropout=self.dropout,
      drop_prob=self.drop_prob,histogram=histogram,l2=True)
    self.reg.append(reg)

    self.sum2 = self.deconv5 + self.conv2
    #d6_shape = [ks1,ks1,self.num_class,num_k2]
    d6_shape = [ks1,ks1,self.num_class,num_k1]
    self.deconv6,reg = ut.deconv2(inp=self.sum2,shape=d6_shape,
      relu=False,name='deconv6',histogram=histogram,l2=True)
    self.reg.append(reg)


    self.pre_logits = self.deconv6

    if verb:
      msg = '\n\t{0} \n\t{1} \n\t{2} \n\t{3} \n\t{4} \n\t{5}'
      msg = msg.format(self.conv1,self.conv2,self.conv3,self.conv4,
                       self.conv5,self.conv6)
      msg += '\n\t{0} \n\t{1} \n\t{2} \n\t{3} \n\t{4} \n\t{5}'
      msg = msg.format(self.deconv1,self.deconv2,self.deconv3,
                       self.deconv4,self.deconv5,self.deconv6)
      print(msg)

  def last_layer(self):
    """
    Returns the last layer of the model
    """
    return(self.pre_logits)

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