# coding=utf-8
"""
Author: Héctor Sánchez
Date: February-13-2018
Description: Contains Main Segment Training Model
"""
import os
import sys
# Hidde debug infro from tf... https://stackoverflow.com/a/38645250/5969548
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import models as md

class SegModel:
  """
  This object will select one model and train it for the purpose of 
  semantic segmentation
  """
  def __init__(self,im_h=28,im_w=28,im_c=1,num_class=11,model=0,
    training=False,train=None,test=None,val=None,bs=1,ex=1,version=1,
    save=False,save_dir=None,save_checkp=None,load=False,load_dir=None,
    load_checkp=None,save_load_same=True,load_step=None,tb_log=False,
    log_dir=None,log_name=None,deacy_steps=5000,max_to_keep=1):
    """
    """
    self.session = tf.Session()
    self.bs = bs
    self.ex = ex

    self.im_h = im_h
    self.im_w = im_w
    self.im_c = im_c
    self.num_class = num_cla5s

    self.inputs()

    if model==0:
      self.model = md.MaxPool()
    elif model==1:
      self.model = md.MaxPoolSC()
    elif model==2:
      self.model = md.Stride()
    elif model==3:
      self.model = md.StrideSC()
    elif model==4:
      self.model = md.Stride2()
    elif model==5:
      self.model = md.Stride2SC()

    self.model.init(inp=self.x,ex=self.ex,num_class=num_class,
      version=version)

    # Specify where to save the model and the tb log
    save_dir,save_checkp = self.model.checkpoint_dir(save_dir,save_checkp)
    load_dir,load_checkp = self.model.checkpoint_dir(load_dir,load_checkp)
    log_dir,log_name = self.model.log_dir(log_dir,log_name)

    self.last_layer = self.model.last_layer()

    self.outputs()

    if self.training:
      self.bs = bs
      #self.lr = lr
      self.global_step = tf.Variable(0,trainable=False)
      self.lr = tf.train.exponential_decay(lr,self.global_step,deacy_steps,0.96,staircase=True)
      self.dropout = dropout
      self.drop_prob = drop_prob
      self.tb_log = tb_log
      self.load_step = load_step
      self.total_it = 0
      self.best_acc = 0

      self.l2 = True

      self.train = train
      self.val = val
      self.test = test

      self.trainable()

      self.summary = tf.summary.merge_all()
      
      if self.tb_log:
        self.tensorboard_log(log_dir,log_name)

    if self.save or self.load:
      self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    self.init_variables()

    if self.load:
      self.loadable(load_dir,load_checkp,save_load_same)

  def predict(self,inp):
    feed_dict = {self.x: inp}
    out = self.session.run(self.y_pred_cls_seg,feed_dict=feed_dict)
    return(out)

  def optimize(self,num_it=0,verb=None,tb_log_it=100):
    """
    Description: Trains the model

    num_it: Number of iterations to train
    verb: Display training process each 'verb' iterations
    tb_log_it: Saves summary each 'tb_log_it' if log is True
    """
    assert self.training, 'Not loaded as a trainable model. Try reloading it as trainable.'
    print('\nStarting optimization...\n')
    for it in range(num_it):
      self.total_it += 1

      data = self.train.next_batch(self.bs)
      
      if self.ex is not None and data['ims'].shape[0]<self.ex:
        # Ensures that won't be an error caused by shape incompatibility
        continue

      if self.dropout:
        feed_dict = {self.x: data['ims'], 
                     self.y_seg: data['seg'],
                     self.drop_prob: self.drop_prob}
      else:
        feed_dict = {self.x: data['ims'], 
                     self.y_seg: data['seg']}

      acc,_ = self.session.run([self.accuracy,self.optimizer],feed_dict=feed_dict)

      if verb is not None and self.total_it%verb==0:
        #acc = self.full_acc(self.val,bs=self.bs)

        if self.best_acc<acc:
          self.best_acc = acc
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
          #acc = self.full_acc(self.val,self.bs)
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
        num_ex -= 1
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
        learning_rate=self.lr).minimize(self.cost,global_step=self.global_step)

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
      tf.summary.image('seg_input',self.seg_inp_vis,1)
      ### END: Seg Image visualization

      msg = '\n\t{0} \n\t{1} \n\t{2} \n\t{3}'
      msg = msg.format(self.x,self.y_seg,self.y_seg_cls,self.y_seg_onehot)
      print(msg)

  def close_session(self):
    self.session.close()
    print('Session closed')

class SegMultiClass(SegModel):
  def __init__(self):
    pass

  def init(self,im_h=28,im_w=28,im_c=1,num_class=11,model=0,
    training=False,train=None,test=None,val=None,bs=1,ex=1,version=1,
    save=False,save_dir=None,save_checkp=None,load=False,load_dir=None,
    load_checkp=None,save_load_same=True,load_step=None,tb_log=False,
    log_dir=None,log_name=None,deacy_steps=10000,version=1,max_to_keep=1):

    SegModel.__init__(self,im_h=im_h,im_w=im_w,im_c=im_c,
      num_class=num_class,model=model,training=training,train=train,
      test=test,val=val,bs=bs,ex=ex,version=version,save=save,
      save_dir=save_dir,save_checkp=save_checkp,load=load,
      load_dir=load_dir,load_checkp=load_checkp,
      save_load_same=save_load_same,load_step=load_step,tb_log=tb_log,
      log_dir=log_dir,log_name=log_name,deacy_steps=deacy_steps,
      max_to_keep=max_to_keep)

class SegBinaryClass(SegModel):
  def __init__(self):
    pass

  def init(self,im_h=28,im_w=28,im_c=1,num_class=11,model=0,
    training=False,train=None,test=None,val=None,bs=1,ex=1,version=1,
    save=False,save_dir=None,save_checkp=None,load=False,load_dir=None,
    load_checkp=None,save_load_same=True,load_step=None,tb_log=False,
    log_dir=None,log_name=None,deacy_steps=5000,version=1,max_to_keep=1):

    SegModel.__init__(self,im_h=im_h,im_w=im_w,im_c=im_c,
      num_class=num_class,model=model,training=training,train=train,
      test=test,val=val,bs=bs,ex=ex,version=version,save=save,
      save_dir=save_dir,save_checkp=save_checkp,load=load,
      load_dir=load_dir,load_checkp=load_checkp,
      save_load_same=save_load_same,load_step=load_step,tb_log=tb_log,
      log_dir=log_dir,log_name=log_name,deacy_steps=deacy_steps,
      max_to_keep=max_to_keep)

  def trainable(self):
    """
    Creates tensors needed for training the model
    """

    # TODO: beta value accessible from somewhere else
    beta = 0.01
    with tf.name_scope('cross_entropy'):
      self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.logits,labels=tf.cast(self.y_seg_cls,tf.float32),
        name='cross_entropy')
      self.cost = tf.reduce_mean(self.cross_entropy,name='cost')
      if self.l2:
        for i,reg in enumerate(self.model.reg):
          self.cost += tf.reduce_mean(beta*reg,name='cost-w'+str(i))
      tf.summary.scalar('cost',self.cost)

    with tf.name_scope('train'):
      self.optimizer = tf.train.AdamOptimizer(
        learning_rate=self.lr).minimize(self.cost,global_step=self.global_step)

      # TODO: Fix batch norm
      ### STARTS: For batch norm... mean and variance
      #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      #
      #with tf.control_dependencies(update_ops):
      #  train_ops = [self.optimizer] + update_ops
      #  train_op_final = tf.group(*train_ops)
      ### ENDS: For batch norm... mean and variance

    with tf.name_scope('accuracy'):
      self.correct_prediction = tf.equal(self.y_pred_cls,
                                         tf.cast(self.y_seg_cls,tf.float32),
                                         name='correct_prediction')
      self.accuracy = tf.reduce_mean(
        tf.cast(self.correct_prediction, tf.float32))
      tf.summary.scalar('accuracy',self.accuracy)
  
  def outputs(self):
    """
    Creates all placeholders needed to predict segmentation
    """
    with tf.name_scope('Prediction'):
      self.last_layer = tf.nn.sigmoid(self.last_layer)
      
      self.y_pred = tf.cast(tf.greater(self.last_layer,0.5),tf.float32)
      self.logits = tf.reshape(self.y_pred,
        shape=[-1,self.num_class],name='logits_rs')

      self.y_pred_cls = tf.reshape(self.y_pred,shape=[-1,1],
        name='y_pred_cls')
      self.y_pred_cls_seg = tf.reshape(self.y_pred,
        shape=[-1,self.im_h,self.im_w,1],name='y_pred_cls_seg')
      
      ### START: Seg Image visualization
      #self.seg_out_vis = tf.divide(self.y_pred_cls_seg,
      #  self.num_class)
      self.seg_out_vis = tf.cast(tf.scalar_mul(255,self.y_pred_cls_seg),
        tf.uint8)
      tf.summary.image('seg_output',self.seg_out_vis,1)
      ### END: Seg Image visualization

      msg = '\n\t{0} \n\t{1} \n\t{2} \n\t{3}'
      msg = msg.format(self.logits,self.y_pred,
              self.y_pred_cls,self.y_pred_cls_seg)
      print(msg)
