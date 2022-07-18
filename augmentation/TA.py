# %matplotlib inline
import torch
import torchvision
from torchvision import transforms
from torch import nn
from d2l import torch as d2l
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
import sys
sys.path.append(r"C:\Users\t-jiahuihe\Code\PythonCode\TestForFirst\augmentation\aug_lib.py")
# from TestForFirst.augmentation import aug_lib
import aug_lib

from PIL import Image
import matplotlib.pyplot as plt
img = Image.open(r'C:\Users\t-jiahuihe\OneDrive\learn\pytorch\chapter_computer-vision\dog.jpg')
plt.figure('image')
# plt.imshow(img)
# plt.show()
print(img.size)# (600, 744)(宽，长)

# aug_lib.set_augmentation_space()
augmenter = aug_lib.TrivialAugment()
aug_img = augmenter(img)
plt.imshow(aug_img)
plt.show()
print(type(aug_img))# <class 'PIL.Image.Image'>