#%%
import torch
from torch import nn
import torchvision as TV
from IPython.display import clear_output

train_data = TV.datasets.MNIST("../dataset/MNIST/", train=True, transform=None,target_transform=None,download=True) #下載並匯入MNIST訓練資料
test_data = TV.datasets.MNIST("../dataset/MNIST/", train=False, transform=None,target_transform=None,download=True) #下載並匯入MNIST測試資料

print('Number of samples in train_data is: ',len(train_data))
print('Number of samples in test_data is: ',len(test_data))   
#%%
from matplotlib import pyplot as plt
import numpy as np
x = train_data.data[0] #讀取訓練集中的第一張圖片
plt.imshow(x) #把圖片顯示出來

# %%
