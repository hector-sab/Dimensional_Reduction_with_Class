"""
Author: Héctor Sánchez
Date: January-31-2018
Description: Contains the different models for MNIST segmentation
"""
import sys
import utils as ut
with ut.HiddenPrints():
  import tensorflow as tf


class SegModel:
  """
  This object will select one model and train it for the purpose of 
  semantic segmentation
  """
  # TODO: Check if visualization nodes can be reused to create just one
  def __init__(self,train,val,test=None,num_class=11,ex=None,model=0,
    bs=1,lr=3e-5,dropout=False,drop_prob=0.25,training=True,save=False,
    save_dir='checkpoints/mnist_seg_mpv1/',save_checkp='mnist_seg',
    max_to_keep=1,load=False,load_dir='checkpoints/mnist_seg_mpv1/',
    load_checkp='mnist_seg',save_load_same=True,load_step=None,
    tb_log=False,log_dir='./log/mnist_seg_v1/',log_name='mnist_seg'):
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

    # Image specs
    self.im_h = self.train.images.shape[1]
    self.im_w = self.train.images.shape[2]
    self.im_c = self.train.images.shape[3]

    self.num_class = num_class

    # Create input placeholders
    self.inputs()

    # Select the model to train
    if model==0:
      pass
    elif model==1:
      self.model = ModelMPv1(inp=self.x,dropout=self.dropout,
        drop_prob=self.drop_prob,histogram=self.tb_log,
        num_class=self.num_class)
    else:
      print("There's no model with that option choice...")
      sys.exit()

    self.last_layer = self.model.last_layer()

    # Create output placeholders
    self.outputs()

    if self.training:
      self.trainable()

    self.summary = tf.summary.merge_all()

    if self.save or self.load:
      self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    self.init_variables()

    if self.save:
      self.savable(save_dir)
    if self.load:
      self.loadable(load_dir,save_load_same)

    if self.tb_log:
      self.tensorboard_log(log_dir,log_name)


  def optimize(self,num_it=0,verb=None,tb_log_it=100):
    """
    Description: Trains the model

    num_it: Number of iterations to train
    verb: Display training process each 'verb' iterations
    tb_log_it: Saves summary each 'tb_log_it' if log is True
    """
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
        acc = self.full_acc(self.val,1)

        if self.best_acc<acc:
          self.best_val_acc = acc
          saved_str = '*'

          if self.save:
            self.saver.save(sess=self.session,save_path=self.save_path,
              global_step=self.total_it)
        else:
          saved_str = ''
        msg = 'It: {0}/{1} - Acc {2:.1%} {3}'.format(self.total_it,
                self.total_it+num_it-(it+1),acc,saved_str)
        print(msg)

      if verb is None and self.save:
        """
        In case verbose is disabled, to ensure checkpoints are saved
        """
        if self.total_it%100==0:
          acc = self.full_acc(self.val,1)
          if self.best_acc<acc:
            self.best_val_acc = acc
            self.saver.save(sess=self.session,save_path=self.save_path,
                global_step=self.total_it)

      if self.tb_log and self.total_it%tb_log_it==0:
        self.val.restart_next_batch()
        data = val.next_batch(self.bs)
        tmp_feed = {self.x:data['ims'], self.y_seg:data['seg']}
        s = self.session.run(self.summary,feed_dict=tmp_feed)
        self.writer.add_summary(s,self.total_it)





  def test_acc(self,bs=1):
    """
    Description: Returns the accuracy on test set
    """
    acc = self.full_acc(self.test,bs)
    return(acc)

  def full_acc(self,dataset,bs):
    """
    Description: Returns the validation accuracy when batch size
            is equal to bs (one...).

    data: DataSet used to calculate full accuracy. It cann be
          train, val, or test
    """
    dataset.restart_next_batch()
    num_ex = dataset.images.shape[0]
    total_acc = 0
    
    for it in range(num_ex):
      data = dataset.next_batch(bs)
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
    msg = 'Saving Tensorboard log at: {0}'.format(log_dir)
    print(msg)
    self.writer.add_graph(self.session.graph)

  def savable(self,save_dir):
    """
    Save path to be saved
    """
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    self.save_path = os.path.join(save_dir,save_checkp)

  def loadable(self,load_dir,save_load_same):
    """
    Load path to be saved
    """
    if save_load_same:
      load_path = self.save_path
    else:
      load_path = os.path.join(load_dir,load_checkp)

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
      self.saver.restore(sess=self.session, save_path=load_path,
        global_step=load_step)
      self.total_it = load_step

  def trainable(self):
    """
    Creates tensors needed for training the model
    """
    with tf.name_scope('cross_entropy'):
      self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=self.logits,labels=self.y_seg_onehot,name='cross_entropy')
      self.cost = tf.reduce_mean(self.cross_entropy,name='cost')
      tf.summary.scalar('cost',self.cost)

    with tf.name_scope('train'):
      self.optimizer = tf.train.AdamOptimizer(
        learning_rate=self.lr).minimize(self.cost)

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
      tf.summary.image('seg output',self.seg_out_vis,1)
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
      self.drop_prob = tf.placeholder_with_default(0.0,shape=[],
        name='drop_prob')
      self.x = tf.placeholder(tf.float32,\
        shape=[self.ex,self.im_h,self.im_w,self.im_c],name='x')
      tf.summary.image('input',self.x,1)

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
  Contains the model of the segmentation using strides of two,
  and no max pooling
  """
  def __init__(self,inp,dropout=False,drop_prob=0.25,
    histogram=True,num_class=11):
    """
    inp: Input placeholder.
    shape: Tensorflow tensor shape used in the input placeholder.
           It must be a list object.
    dropout: Flag used to indicate if dropout will be used
    drop_prob: Percentage of neurons to be turned off
    histogram: Indicates if information for tensorboard should be annexed.
    """
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
    self.conv1 = ut.conv2(inp=self.x,shape=c1_shape,name='conv1',
      dropout=self.dropout,drop_prob=self.drop_prob,histogram=histogram)
    
    c2_shape = [ks2,ks2,num_k1,num_k2]
    self.conv2 = ut.conv2(inp=self.conv1,shape=c2_shape,
      name='conv2',dropout=self.dropout,drop_prob=self.drop_prob,
      histogram=histogram)

    self.pool1,self.ind1 = ut.max_pool(self.conv2,args=True,
      name='maxpool1')

    c3_shape = [ks3,ks3,num_k2,num_k3]
    self.conv3 = ut.conv2(inp=self.pool1,shape=c3_shape,name='conv3',
      dropout=self.dropout,drop_prob=self.drop_prob,histogram=histogram)
    
    c4_shape = [ks4,ks4,num_k3,num_k4]
    self.conv4 = ut.conv2(inp=self.conv3,shape=c4_shape,name='conv4',
      dropout=self.dropout,drop_prob=self.drop_prob,histogram=histogram)

    self.pool2,self.ind2 = ut.max_pool(self.conv4,args=True,
      name='maxpool2')

    c5_shape = [ks5,ks5,num_k4,num_k5]
    self.conv5 = ut.conv2(inp=self.pool2,shape=c5_shape,name='conv5',
      dropout=self.dropout,drop_prob=self.drop_prob,histogram=histogram)

    c6_shape = [ks6,ks6,num_k5,num_k6]
    self.conv6 = ut.conv2(inp=self.conv5,shape=c6_shape,name='conv6',
      dropout=self.dropout,drop_prob=self.drop_prob,histogram=histogram)



    d1_shape = [ks6,ks6,num_k5,num_k6]
    self.deconv1 = ut.deconv2(inp=self.conv6,shape=d1_shape,
      relu=True,name='deconv1',dropout=self.dropout,
      do_prob=self.drop_prob)
    d2_shape = [ks5,ks5,num_k4,num_k5]
    self.deconv2 = ut.deconv2(inp=self.deconv1,shape=d2_shape,
      relu=True,name='deconv2',dropout=self.dropout,
      do_prob=self.drop_prob)
    self.unpool1 = ut.unpool_with_argmax(self.deconv2,self.ind2,
                      input_shape=[1,7,7,num_k4],name='unpool1')

    self.sum1 = self.unpool1 + self.conv4
    d3_shape = [ks4,ks4,num_k3,num_k4]
    self.deconv3 = ut.deconv2(inp=self.sum1,shape=d3_shape,
      relu=True,name='deconv3',dropout=self.dropout,
      do_prob=self.drop_prob)
    d4_shape = [ks3,ks3,num_k2,num_k3]
    self.deconv4 = ut.deconv2(inp=self.deconv3,shape=d4_shape,
      relu=True,name='deconv4',dropout=self.dropout,
      do_prob=self.drop_prob)
    self.unpool2 = ut.unpool_with_argmax(self.deconv4,self.ind1,
                      input_shape=[1,14,14,num_k2],name='unpool2')

    self.sum2 = self.unpool2 + self.conv2
    d5_shape = [ks2,ks2,num_k2,num_k2]
    self.deconv5 = ut.deconv2(inp=self.sum2,shape=d5_shape,
      relu=True,name='deconv5',dropout=self.dropout,
      do_prob=self.drop_prob)

    d6_shape = [ks1,ks1,self.num_class,num_k2]
    self.deconv6 = ut.deconv2(inp=self.deconv5,shape=d6_shape,
                   relu=False,name='deconv6')
    """
    outlike6 = tf.placeholder(tf.float32,
      shape=[self.ex,28,28,self.num_seg_class],name='d6_tmp')
    self.deconv6 = ut.deconv(inp=self.deconv5,out_like=__outlike6,
      shape=__d6_shape,relu=False,pr=False,name='deconv6')
    """
    self.pre_logits = self.deconv6

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