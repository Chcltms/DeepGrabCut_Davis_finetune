#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from torch.utils.data import Dataset


# In[9]:


class Davis_refine_loader(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        
        self.train_file = '/home/ningxinLin/DeepGrabCut_Davis_finetune/data/Davis2017/train/train.txt'
        self.train_list = self.read_list(self.train_file)
        self.transform = transform
        #print(self.train_list)
        
    def __getitem__(self, index):
            #__img_path, __target_path = self.train_list[index].strip().split(' ')
        path = self.train_list[index]
        _img_path, _target_path = path.strip().split(' ')
        _img = cv2.imread(_img_path)
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        _target = cv2.imread(_target_path, 0)
        __categorys = np.unique(_target)
        for category in __categorys:
            if category != 0:
                __class_num = category
        _target = _target/__class_num
            
        sample = {'image': _img, 'gt':_target}
            
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
        
        
    def read_list(self, path):
        f = open(self.train_file, 'r')
        lines = f.readlines()
        f.close()
        return lines

    def __len__(self):
        return len(self.train_list)


# In[10]:





# In[ ]:





# In[ ]:




