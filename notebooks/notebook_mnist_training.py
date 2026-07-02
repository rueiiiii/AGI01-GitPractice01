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
class CNN(torch.nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1,16,kernel_size=(3,3)) #第一卷積層
    self.conv2 = nn.Conv2d(16,32,kernel_size=(3,3))
    self.maxpool = nn.MaxPool2d(kernel_size=(2,2)) #最大池化層
    self.lin1 = nn.Linear(800,128)
    self.out = nn.Linear(128,10) #模型的最後一層為線性層，用來處理CNN扁平化後的輸出
  def forward(self,x):
    x = self.conv1(x)
    x = nn.functional.relu(x) #選用ReLU為激活函數
    x = self.maxpool(x)
    x = self.conv2(x)
    x = nn.functional.relu(x)
    x = self.maxpool(x)
    x = x.flatten(start_dim=1) #扁平化
    x = self.lin1(x) 
    x = nn.functional.relu(x)
    x = self.out(x)
    x = nn.functional.log_softmax(x,dim=1) #使用log_softmax( )，以機率的形式進行分類
    return x
# %%
