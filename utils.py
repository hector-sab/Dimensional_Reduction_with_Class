import os
import sys
import numpy as np


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
  
  # TODO: Fix axes.flat[i] when ncols and nrows = 1
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

def vali_end_path(path_name):
  """
  Checks that the end of the path name contains a '/'
  """
  if path_name[-1]!='/':
    path_name += '/'
  return(path_name)