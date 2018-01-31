"""
Author: Héctor Sánchez
Date: January-30-2018
Description: model for MNIST segmentation
"""
import tensorflow as tf
import utils as ut
import os

class SegModel:
  def __init__(self,train,val,test=None,num_class=10,ex=1,lr=3e-7,
    save=False,save_dir='checkpoints/mnist_seg/',save_checkp='mnist_seg',
    load=False,load_dir='checkpoints/mnist_seg/',load_checkp='mnist_seg',
    save_load_same=True,load_step=None,log=False,log_dir='./log/mnist_seg_v1/',
    log_name='mnist_seg',dropout=False,do_prob=0.25):
    """
    -train: Triaining data using the class DataSeg
    -val: Validation data using the class DataSeg
    -test: Test data using the class DataSeg
    -num_class: Number of classes. I.E. MNIST has 10 classes
    -ex: Number of examples used to train the model
    -lr: Learning rate
    -save: Flag indicating if the model will save checkpoints
    -save_dir: Directory where the model will be saved
    -save_checkp: Name of the checkpoint to be saved
    -load: Flag indicating if the model will load checkpoints
    -load_dir: Directory where the pre-trained model is located
    -load_checkp: Name of the checkpoint to be loaded
    -save_load_same: Flag indicating if the load and save model 
              are the same
    -load_step: if the model as many checkpoints steps, select which
              one should be loaded
    -log: Flag indicating if summaries will be saved
    -log_dir: Directory where it will be saved
    -log_name: Name of the summary
    """
    # Data
    self.train = train
    self.val = val
    self.test = test

    # Image specs
    self.ex = ex # Number of examples
    self.bs = self.ex
    self.im_h = train.images.shape[1]
    self.im_w = train.images.shape[2]
    self.im_c = train.images.shape[3]
    self.num_class = num_class
    self.num_seg_class = self.num_class + 1

    self.lr = lr
    self.save = save
    self.load = load
    self.load_step = load_step
    self.log = log

    self.dropout = dropout
    self.do_prob = do_prob

    self.total_it = 0
    self.best_val_acc = 0

    self.session = tf.Session()

    # Input data
    msg = '\nInitializing model...'
    print(msg)

    with tf.name_scope('Input'):
      self.drop_prob = tf.placeholder_with_default(0.0,shape=[])
      # Normal image
      self.x = tf.placeholder(tf.float32,\
        shape=[self.ex,self.im_h,self.im_w,self.im_c], name='x')
      tf.summary.image('input',self.x,1)

      # Image with pixel-lebel labels
      self.y_seg = tf.placeholder(tf.int64,\
        shape=[self.ex,self.im_h,self.im_w,1],name='y_seg')
      self.y_seg_cls = tf.reshape(self.y_seg,shape=[-1,1],\
        name='y_seg_cls')

      # Image with 'num_seg_class' channels
      self.y_seg_onehot = tf.one_hot(self.y_seg_cls,\
        depth=self.num_seg_class,axis=1)
      # Reshape to errase an useless extra dimension
      self.y_seg_onehot = tf.reshape(self.y_seg_onehot,\
        shape=[-1,self.num_seg_class],name='y_seg_onehot')

      ### START: Seg Image visualization
      self.seg_inp_vis = tf.divide(self.y_seg,self.num_seg_class)
      self.seg_inp_vis = tf.cast(tf.scalar_mul(255,self.seg_inp_vis),
        tf.uint8)
      tf.summary.image('seg input',self.seg_inp_vis,1)
      ### END: Seg Image visualization

      msg = '\n\t{0} \n\t{1} \n\t{2} \n\t{3}'
      msg = msg.format(self.x,self.y_seg,self.y_seg_cls,self.y_seg_onehot)
      print(msg)

    self.model = self.__model()

    with tf.name_scope('Prediction'):
      self.logits = tf.reshape(self.pre_logits,
        shape=[-1,self.num_seg_class],name='logits_rs')
      self.y_pred = tf.nn.softmax(self.logits,name='y_pred')
      self.y_pred_cls = tf.argmax(self.y_pred,axis=1)
      self.y_pred_cls = tf.reshape(self.y_pred_cls,shape=[-1,1],
        name='y_pred_cls')
      self.y_pred_cls_seg = tf.reshape(self.y_pred_cls,
        shape=[self.ex,self.im_h,self.im_w,1],name='y_pred_cls_seg')

      ### START: Seg Image visualization
      self.seg_out_vis = tf.divide(self.y_pred_cls_seg,
        self.num_seg_class)
      self.seg_out_vis = tf.cast(tf.scalar_mul(255,self.seg_out_vis),
        tf.uint8)
      tf.summary.image('seg output',self.seg_out_vis,1)
      ### END: Seg Image visualization

      msg = '\n\t{0} \n\t{1} \n\t{2} \n\t{3}'
      msg = msg.format(self.logits,self.y_pred,
              self.y_pred_cls,self.y_pred_cls_seg)
      print(msg)

    with tf.name_scope('cross_entropy'):
      self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=self.logits,labels=self.y_seg_onehot)
      self.cost = tf.reduce_mean(self.cross_entropy)
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

    # TODO: Check if it works after init_variables
    self.summary = tf.summary.merge_all()



    # Save/Load checkpoints
    if self.save:
      self.saver = tf.train.Saver(max_to_keep=5000)
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      self.save_path = os.path.join(save_dir,save_checkp)
    
    if self.load:
      if not self.save:
        self.saver = tf.train.Saver()
      if save_load_same:
        self.load = True
        self.load_path = self.save_path
      else:
        self.load_path = os.path.join(load_dir,load_checkp)

      # TODO: check if restore_vars works before init_var
      #self.restore_variables()

    
    self.init_variables()

    if self.load:
      self.restore_variables()

    if self.log:
      if not os.path.exists(log_dir):
        os.makedirs(log_dir)
      self.writer = tf.summary.FileWriter(log_dir+log_name)
      msg = log_dir
      self.writer.add_graph(self.session.graph)

    val_data  = self.val.next_batch(self.bs,seg_fb=False)
    self.feed_val = {self.x:val_data['ims'], self.y_seg:val_data['seg']}

  def init_variables(self):
    self.session.run(tf.global_variables_initializer())

  def restore_variables(self):
    if self.load_step is None:
      self.saver.restore(sess=self.session, save_path=self.load_path)
    else:
      self.saver.restore(sess=self.session, save_path=self.load_path,
        global_step=self.load_step)
      self.total_it = self.load_step

  def __model(self):
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

    # Core model
    c1_shape = [ks1,ks1,self.im_c,num_k1]
    self.conv1 = ut.conv(inp=self.x,shape=c1_shape,name='conv1',
      dropout=self.dropout,do_prob=self.drop_prob)
    c2_shape = [ks2,ks2,num_k1,num_k2]
    self.conv2 = ut.conv(inp=self.conv1,shape=c2_shape,
      name='conv2',dropout=self.dropout,
      do_prob=self.drop_prob)
    self.pool1,self.pool1_args = ut.pool_argmax(self.conv2)

    c3_shape = [ks3,ks3,num_k2,num_k3]
    self.conv3 = ut.conv(inp=self.pool1,shape=c3_shape,name='conv3',
      dropout=self.dropout,do_prob=self.drop_prob)
    c4_shape = [ks4,ks4,num_k3,num_k4]
    self.conv4 = ut.conv(inp=self.conv3,shape=c4_shape,
      strides=[1,2,2,1],name='conv4',dropout=self.dropout,
      do_prob=self.drop_prob)
    self.pool2,self.pool2_args = ut.pool_argmax(self.conv4)

    c5_shape = [ks5,ks5,num_k4,num_k5]
    self.conv5 = ut.conv(inp=self.pool2,shape=c5_shape,name='conv5',
      dropout=self.dropout,do_prob=self.drop_prob)
    c6_shape = [ks6,ks6,num_k5,num_k6]
    self.conv6 = ut.conv(inp=self.conv5,shape=c6_shape,name='conv6',
      dropout=self.dropout,do_prob=self.drop_prob)

    d1_shape = [ks6,ks6,num_k5,num_k6]
    self.deconv1 = ut.deconv2(inp=self.conv6,relu=True,
      shape=d1_shape,dropout=self.dropout,
      do_prob=self.drop_prob)
    d2_shape = [ks5,ks5,num_k4,num_k5]
    self.deconv2 = ut.deconv2(inp=self.deconv1,shape=d2_shape,
      relu=True,name='deconv2',dropout=self.dropout,
      do_prob=self.drop_prob)
    self.unpool1 = ut.unpool_with_argmax(self.deconv2,self.pool2_args,
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
    self.unpool2 = ut.unpool_with_argmax(self.deconv4,self.pool1_args,
                      input_shape=[1,14,14,num_k2],name='unpool2')

    self.sum2 = self.unpool2 + self.conv2
    d5_shape = [ks2,ks2,num_k2,num_k2]
    self.deconv5 = ut.deconv2(inp=self.sum2,shape=d5_shape,
      relu=True,name='deconv5',dropout=self.dropout,
      do_prob=self.drop_prob)

    d6_shape = [ks1,ks1,self.num_seg_class,num_k2]
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

  def save_checkpoints(self,save=False):
    if save:
      self.save = True

  def optimize(self,num_it=0,print_it=100,log_it=100,
    print_test_acc=False,print_test_it=1000):
    """
    num_it: Number of iterations to train
    print_it: Print partial accuracy from val at 'print_it'
    log_it: Saves summary at 'log_it' if log is True
    print_test_acc: Print test accuracy
    print_test_it: Print test accuracy at 'print_test_it'
    """

    for i in range(num_it):
      self.total_it += 1
      data = self.train.next_batch(self.bs)
      #print(self.total_it,data['ims'].shape,data['seg'].shape)
      if self.dropout:
        feed_dict = {self.x:data['ims'],self.y_seg:data['seg'],
                     self.drop_prob:self.do_prob}
      else:
        feed_dict = {self.x:data['ims'],self.y_seg:data['seg']}

      pred_seg,_ = self.session.run([self.y_pred_cls,self.optimizer],
                   feed_dict=feed_dict)

      if self.total_it%log_it==0 and self.log:
        s = self.session.run(self.summary,feed_dict=self.feed_val)
        self.writer.add_summary(s,self.total_it)

      if self.total_it%print_it==0:
        acc = self.session.run(self.accuracy,feed_dict=self.feed_val)
        
        if self.best_val_acc<acc:
          self.best_val_acc = acc
          saved_str = '*'
          
          if self.save:
            self.saver.save(sess=self.session,save_path=self.save_path,
              global_step=self.total_it)
        else:
          saved_str = ''
          
        print('It: {0} - Acc: {1:.1%} {2}'.format(self.total_it,
          acc,saved_str))


      if print_test_acc and self.total_it%print_test_it==0:
        test_acc = self.test_accuracy()
        print('--->Test Acc: {0: .1%}'.format(test_acc))
  
  def predict(self,im):
    """
    im: Contains the images to be evaluated. Shape [num,im_h,im_w,im_c]
    """
    feed_dict = {self.x:im}
    pred = self.session.run(self.y_pred_cls_seg,feed_dict=feed_dict)
    return(pred)

  def test_accuracy(self):
    """
    Returns the test accuracy
    """
    self.test.restart_next_batch()
    num_ex = self.test.images.shape[0]
    total_acc = 0
    for i in range(num_ex):
      data = self.test.next_batch(self.bs)
      feed_dict = {self.x:data['ims'],self.y_seg:data['seg']}
      acc = self.session.run(self.accuracy,feed_dict=feed_dict)
      total_acc += acc

    total_acc /= num_ex
    return(total_acc)