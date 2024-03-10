import torch
import os
from enum import Enum
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as dset
import os, sys, hashlib, torch
import sys
import numpy as np
if sys.version_info[0] == 2:
  import cPickle as pickle
else:
  import pickle
  
  
NORMALIZERS = {
    'cifar10': ([x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]]),
    'cifar100': ([x / 255 for x in [129.3, 124.1, 112.4]],[x / 255 for x in [68.2, 65.4, 70.4]]),
    'imagenet': ([x / 255 for x in [122.68, 116.66, 104.01]], [x / 255 for x in [63.22,  61.26 , 65.09]])
}

def get_dataset(data, root):
  if data in  ['cifar10','cifar100']:
    lists_train = [
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
        ]
    
    lists_val = [transforms.ToTensor()]
    
    train_transform = transforms.Compose(lists_train)
    valid_transform = transforms.Compose(lists_val)
    
    if data == 'cifar10':
        train_data = dset.CIFAR10(root=root, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=root, train=False, download=True, transform=valid_transform)
        num_classes = 10
        
    elif data == 'cifar100':
        train_data = dset.CIFAR100(root=root, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=root, train=False, download=True, transform=valid_transform)
        num_classes = 100
    
  elif data == 'imagenet':
    lists_train = [
        transforms.RandomCrop(16, padding=2), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor()
      ]
    lists_val = [transforms.ToTensor()]
    
    train_transform = transforms.Compose(lists_train)
    valid_transform  = transforms.Compose(lists_val)
    train_data = ImageNet16(os.path.join(root,"ImageNet16"), True , train_transform, 120)
    valid_data  = ImageNet16(os.path.join(root,"ImageNet16"), False, valid_transform , 120)
    num_classes = 120
    assert len(train_data) == 151700 and len(valid_data) == 6000
  else:
    NotImplementedError

  return train_data,valid_data,num_classes



class Stage(Enum):
    train = 1
    val = 2
    test = 3


class ImageNet16(torch.utils.data.Dataset):
  """https://github.com/D-X-Y/AutoDL-Projects/blob/f46486e21b71ae6459a700be720d7648b5429569/xautodl/datasets/DownsampledImageNet.py#L36"""
  # http://image-net.org/download-images
  # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
  # https://arxiv.org/pdf/1707.08819.pdf

  train_list = [
        ['train_data_batch_1', '27846dcaa50de8e21a7d1a35f30f0e91'],
        ['train_data_batch_2', 'c7254a054e0e795c69120a5727050e3f'],
        ['train_data_batch_3', '4333d3df2e5ffb114b05d2ffc19b1e87'],
        ['train_data_batch_4', '1620cdf193304f4a92677b695d70d10f'],
        ['train_data_batch_5', '348b3c2fdbb3940c4e9e834affd3b18d'],
        ['train_data_batch_6', '6e765307c242a1b3d7d5ef9139b48945'],
        ['train_data_batch_7', '564926d8cbf8fc4818ba23d2faac7564'],
        ['train_data_batch_8', 'f4755871f718ccb653440b9dd0ebac66'],
        ['train_data_batch_9', 'bb6dd660c38c58552125b1a92f86b5d4'],
        ['train_data_batch_10','8f03f34ac4b42271a294f91bf480f29b'],
    ]
  valid_list = [
        ['val_data', '3410e3017fdaefba8d5073aaa65e4bd6'],
    ]
  
  def __init__(self, root, train, transform, use_num_of_class_only=None):
    self.root      = root
    self.transform = transform
    self.train     = train  # training set or valid set
    if not self._check_integrity(): raise RuntimeError('Dataset not found or corrupted.')

    if self.train: downloaded_list = self.train_list
    else         : downloaded_list = self.valid_list
    self.data    = []
    self.targets = []

    # now load the picked numpy arrays
    for i, (file_name, checksum) in enumerate(downloaded_list):
      file_path = os.path.join(self.root, file_name)
      #print ('Load {:}/{:02d}-th : {:}'.format(i, len(downloaded_list), file_path))
      with open(file_path, 'rb') as f:
        if sys.version_info[0] == 2:
          entry = pickle.load(f)
        else:
          entry = pickle.load(f, encoding='latin1')
        self.data.append(entry['data'])
        self.targets.extend(entry['labels'])
    self.data = np.vstack(self.data).reshape(-1, 3, 16, 16)
    self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    if use_num_of_class_only is not None:
      assert isinstance(use_num_of_class_only, int) and use_num_of_class_only > 0 and use_num_of_class_only < 1000, 'invalid use_num_of_class_only : {:}'.format(use_num_of_class_only)
      new_data, new_targets = [], []
      for I, L in zip(self.data, self.targets):
        if 1 <= L <= use_num_of_class_only:
          new_data.append( I )
          new_targets.append( L )
      self.data    = new_data
      self.targets = new_targets
    #    self.mean.append(entry['mean'])
    #self.mean = np.vstack(self.mean).reshape(-1, 3, 16, 16)
    #self.mean = np.mean(np.mean(np.mean(self.mean, axis=0), axis=1), axis=1)
    #print ('Mean : {:}'.format(self.mean))
    #temp      = self.data - np.reshape(self.mean, (1, 1, 1, 3))
    #std_data  = np.std(temp, axis=0)
    #std_data  = np.mean(np.mean(std_data, axis=0), axis=0)
    #print ('Std  : {:}'.format(std_data))

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index] - 1

    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    return img, target

  def __len__(self):
    return len(self.data)

  def _check_integrity(self):
    root = self.root
    for fentry in (self.train_list + self.valid_list):
      filename, md5 = fentry[0], fentry[1]
      fpath = os.path.join(root, filename)
      if not check_integrity(fpath, md5):
        return False
    return True

def check_md5(fpath, md5, **kwargs):
  return md5 == calculate_md5(fpath, **kwargs)

def check_integrity(fpath, md5=None):
  if not os.path.isfile(fpath): return False
  if md5 is None: return True
  else          : return check_md5(fpath, md5)
  
def calculate_md5(fpath, chunk_size=1024 * 1024):
  md5 = hashlib.md5()
  with open(fpath, 'rb') as f:
    for chunk in iter(lambda: f.read(chunk_size), b''):
      md5.update(chunk)
  return md5.hexdigest()