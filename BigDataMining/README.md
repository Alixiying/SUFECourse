# 环境准备

推荐在linux或windows10/11系统准备环境，如果是win11可以考虑wsl2，详细请参考https://learn.microsoft.com/en-us/windows/wsl/install

1. 安装conda或miniconda
```
详细请参考https://anaconda.org.cn/anaconda/install/
```

2. 新建conda环境
```
CMD/POWERSHELL:
conda create --name your_env_name python=3.8
conda activate your_env_name
```
3. 安装PyTorch、CUDA (可选)

CPU版:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

GPU版：

Nvidia 40系显卡安装PyTorch2.0+CUDA11.8
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Nvidia 30系显卡安装PyTorch1.13.1+CUDA11.6
```
conda install pytorch==1.13.1 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```

GPU版最后需在NVIDIA官网上寻找对应的CUDA版本驱动安装，详细见https://developer.nvidia.com/cuda-toolkit-archive

# DenseNet架构

## 1. Torch的基本神经网络模块

* 卷积层
```{python}
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
```
| 参数 | 解释|
|:--- |:--- |
|in_channels | 输入通道数|
|out_channels | 输出通道数|
|kernel_size | 卷积核大小|
|stride | 步长|
|padding | 填充|
|dilation | 卷积核每个点之间的间隔|
|groups | 分组卷积|

*  批量归一化层
```{python}
torch.nn.BatchNorm2d(Channels)
```
在一个Batch内对里面的元素归一化

* ReLU激活函数
```{python}
torch.nn.function.relu(tensor)
```
$$
ReLU(x) = max(0, x)
$$

* 池化层
```{python}
torch.nn.function.avg_pool2d(tensor, kernel_size)
```
> 1x224x224 -> 1x(224//kernel_size)x(224//kernel_size) 

## 2. DenseNet


```python
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary
import math

# 单层稠密连接
class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

# Bottleneck层
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

# Compression 
class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


# DenseNet
class DenseNet(nn.Module):
    def __init__(self, input_size, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        self.input_size = input_size
        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), self.input_size // 4))
        out = F.log_softmax(self.fc(out))
        return out
```

DenseNet-BC(k=12, depth=100)
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [1, 24, 32, 32]             648
       BatchNorm2d-2            [1, 24, 32, 32]              48
            Conv2d-3            [1, 48, 32, 32]           1,152
       BatchNorm2d-4            [1, 48, 32, 32]              96
            Conv2d-5            [1, 12, 32, 32]           5,184
        Bottleneck-6            [1, 36, 32, 32]               0
       BatchNorm2d-7            [1, 36, 32, 32]              72
            Conv2d-8            [1, 48, 32, 32]           1,728
       BatchNorm2d-9            [1, 48, 32, 32]              96
           Conv2d-10            [1, 12, 32, 32]           5,184
       Bottleneck-11            [1, 48, 32, 32]               0
      BatchNorm2d-12            [1, 48, 32, 32]              96
           Conv2d-13            [1, 48, 32, 32]           2,304
      BatchNorm2d-14            [1, 48, 32, 32]              96
           Conv2d-15            [1, 12, 32, 32]           5,184
       Bottleneck-16            [1, 60, 32, 32]               0
      BatchNorm2d-17            [1, 60, 32, 32]             120
           Conv2d-18            [1, 48, 32, 32]           2,880
      BatchNorm2d-19            [1, 48, 32, 32]              96
           Conv2d-20            [1, 12, 32, 32]           5,184
       Bottleneck-21            [1, 72, 32, 32]               0
      BatchNorm2d-22            [1, 72, 32, 32]             144
           Conv2d-23            [1, 48, 32, 32]           3,456
      BatchNorm2d-24            [1, 48, 32, 32]              96
           Conv2d-25            [1, 12, 32, 32]           5,184
       Bottleneck-26            [1, 84, 32, 32]               0
      BatchNorm2d-27            [1, 84, 32, 32]             168
           Conv2d-28            [1, 48, 32, 32]           4,032
      BatchNorm2d-29            [1, 48, 32, 32]              96
           Conv2d-30            [1, 12, 32, 32]           5,184
       Bottleneck-31            [1, 96, 32, 32]               0
      BatchNorm2d-32            [1, 96, 32, 32]             192
           Conv2d-33            [1, 48, 32, 32]           4,608
      BatchNorm2d-34            [1, 48, 32, 32]              96
           Conv2d-35            [1, 12, 32, 32]           5,184
       Bottleneck-36           [1, 108, 32, 32]               0
      BatchNorm2d-37           [1, 108, 32, 32]             216
           Conv2d-38            [1, 48, 32, 32]           5,184
      BatchNorm2d-39            [1, 48, 32, 32]              96
           Conv2d-40            [1, 12, 32, 32]           5,184
       Bottleneck-41           [1, 120, 32, 32]               0
      BatchNorm2d-42           [1, 120, 32, 32]             240
           Conv2d-43            [1, 48, 32, 32]           5,760
      BatchNorm2d-44            [1, 48, 32, 32]              96
           Conv2d-45            [1, 12, 32, 32]           5,184
       Bottleneck-46           [1, 132, 32, 32]               0
      BatchNorm2d-47           [1, 132, 32, 32]             264
           Conv2d-48            [1, 48, 32, 32]           6,336
      BatchNorm2d-49            [1, 48, 32, 32]              96
           Conv2d-50            [1, 12, 32, 32]           5,184
       Bottleneck-51           [1, 144, 32, 32]               0
      BatchNorm2d-52           [1, 144, 32, 32]             288
           Conv2d-53            [1, 48, 32, 32]           6,912
      BatchNorm2d-54            [1, 48, 32, 32]              96
           Conv2d-55            [1, 12, 32, 32]           5,184
       Bottleneck-56           [1, 156, 32, 32]               0
      BatchNorm2d-57           [1, 156, 32, 32]             312
           Conv2d-58            [1, 48, 32, 32]           7,488
      BatchNorm2d-59            [1, 48, 32, 32]              96
           Conv2d-60            [1, 12, 32, 32]           5,184
       Bottleneck-61           [1, 168, 32, 32]               0
      BatchNorm2d-62           [1, 168, 32, 32]             336
           Conv2d-63            [1, 48, 32, 32]           8,064
      BatchNorm2d-64            [1, 48, 32, 32]              96
           Conv2d-65            [1, 12, 32, 32]           5,184
       Bottleneck-66           [1, 180, 32, 32]               0
      BatchNorm2d-67           [1, 180, 32, 32]             360
           Conv2d-68            [1, 48, 32, 32]           8,640
      BatchNorm2d-69            [1, 48, 32, 32]              96
           Conv2d-70            [1, 12, 32, 32]           5,184
       Bottleneck-71           [1, 192, 32, 32]               0
      BatchNorm2d-72           [1, 192, 32, 32]             384
           Conv2d-73            [1, 48, 32, 32]           9,216
      BatchNorm2d-74            [1, 48, 32, 32]              96
           Conv2d-75            [1, 12, 32, 32]           5,184
       Bottleneck-76           [1, 204, 32, 32]               0
      BatchNorm2d-77           [1, 204, 32, 32]             408
           Conv2d-78            [1, 48, 32, 32]           9,792
      BatchNorm2d-79            [1, 48, 32, 32]              96
           Conv2d-80            [1, 12, 32, 32]           5,184
       Bottleneck-81           [1, 216, 32, 32]               0
      BatchNorm2d-82           [1, 216, 32, 32]             432
           Conv2d-83           [1, 108, 32, 32]          23,328
       Transition-84           [1, 108, 16, 16]               0
      BatchNorm2d-85           [1, 108, 16, 16]             216
           Conv2d-86            [1, 48, 16, 16]           5,184
      BatchNorm2d-87            [1, 48, 16, 16]              96
           Conv2d-88            [1, 12, 16, 16]           5,184
       Bottleneck-89           [1, 120, 16, 16]               0
      BatchNorm2d-90           [1, 120, 16, 16]             240
           Conv2d-91            [1, 48, 16, 16]           5,760
      BatchNorm2d-92            [1, 48, 16, 16]              96
           Conv2d-93            [1, 12, 16, 16]           5,184
       Bottleneck-94           [1, 132, 16, 16]               0
      BatchNorm2d-95           [1, 132, 16, 16]             264
           Conv2d-96            [1, 48, 16, 16]           6,336
      BatchNorm2d-97            [1, 48, 16, 16]              96
           Conv2d-98            [1, 12, 16, 16]           5,184
       Bottleneck-99           [1, 144, 16, 16]               0
     BatchNorm2d-100           [1, 144, 16, 16]             288
          Conv2d-101            [1, 48, 16, 16]           6,912
     BatchNorm2d-102            [1, 48, 16, 16]              96
          Conv2d-103            [1, 12, 16, 16]           5,184
      Bottleneck-104           [1, 156, 16, 16]               0
     BatchNorm2d-105           [1, 156, 16, 16]             312
          Conv2d-106            [1, 48, 16, 16]           7,488
     BatchNorm2d-107            [1, 48, 16, 16]              96
          Conv2d-108            [1, 12, 16, 16]           5,184
      Bottleneck-109           [1, 168, 16, 16]               0
     BatchNorm2d-110           [1, 168, 16, 16]             336
          Conv2d-111            [1, 48, 16, 16]           8,064
     BatchNorm2d-112            [1, 48, 16, 16]              96
          Conv2d-113            [1, 12, 16, 16]           5,184
      Bottleneck-114           [1, 180, 16, 16]               0
     BatchNorm2d-115           [1, 180, 16, 16]             360
          Conv2d-116            [1, 48, 16, 16]           8,640
     BatchNorm2d-117            [1, 48, 16, 16]              96
          Conv2d-118            [1, 12, 16, 16]           5,184
      Bottleneck-119           [1, 192, 16, 16]               0
     BatchNorm2d-120           [1, 192, 16, 16]             384
          Conv2d-121            [1, 48, 16, 16]           9,216
     BatchNorm2d-122            [1, 48, 16, 16]              96
          Conv2d-123            [1, 12, 16, 16]           5,184
      Bottleneck-124           [1, 204, 16, 16]               0
     BatchNorm2d-125           [1, 204, 16, 16]             408
          Conv2d-126            [1, 48, 16, 16]           9,792
     BatchNorm2d-127            [1, 48, 16, 16]              96
          Conv2d-128            [1, 12, 16, 16]           5,184
      Bottleneck-129           [1, 216, 16, 16]               0
     BatchNorm2d-130           [1, 216, 16, 16]             432
          Conv2d-131            [1, 48, 16, 16]          10,368
     BatchNorm2d-132            [1, 48, 16, 16]              96
          Conv2d-133            [1, 12, 16, 16]           5,184
      Bottleneck-134           [1, 228, 16, 16]               0
     BatchNorm2d-135           [1, 228, 16, 16]             456
          Conv2d-136            [1, 48, 16, 16]          10,944
     BatchNorm2d-137            [1, 48, 16, 16]              96
          Conv2d-138            [1, 12, 16, 16]           5,184
      Bottleneck-139           [1, 240, 16, 16]               0
     BatchNorm2d-140           [1, 240, 16, 16]             480
          Conv2d-141            [1, 48, 16, 16]          11,520
     BatchNorm2d-142            [1, 48, 16, 16]              96
          Conv2d-143            [1, 12, 16, 16]           5,184
      Bottleneck-144           [1, 252, 16, 16]               0
     BatchNorm2d-145           [1, 252, 16, 16]             504
          Conv2d-146            [1, 48, 16, 16]          12,096
     BatchNorm2d-147            [1, 48, 16, 16]              96
          Conv2d-148            [1, 12, 16, 16]           5,184
      Bottleneck-149           [1, 264, 16, 16]               0
     BatchNorm2d-150           [1, 264, 16, 16]             528
          Conv2d-151            [1, 48, 16, 16]          12,672
     BatchNorm2d-152            [1, 48, 16, 16]              96
          Conv2d-153            [1, 12, 16, 16]           5,184
      Bottleneck-154           [1, 276, 16, 16]               0
     BatchNorm2d-155           [1, 276, 16, 16]             552
          Conv2d-156            [1, 48, 16, 16]          13,248
     BatchNorm2d-157            [1, 48, 16, 16]              96
          Conv2d-158            [1, 12, 16, 16]           5,184
      Bottleneck-159           [1, 288, 16, 16]               0
     BatchNorm2d-160           [1, 288, 16, 16]             576
          Conv2d-161            [1, 48, 16, 16]          13,824
     BatchNorm2d-162            [1, 48, 16, 16]              96
          Conv2d-163            [1, 12, 16, 16]           5,184
      Bottleneck-164           [1, 300, 16, 16]               0
     BatchNorm2d-165           [1, 300, 16, 16]             600
          Conv2d-166           [1, 150, 16, 16]          45,000
      Transition-167             [1, 150, 8, 8]               0
     BatchNorm2d-168             [1, 150, 8, 8]             300
          Conv2d-169              [1, 48, 8, 8]           7,200
     BatchNorm2d-170              [1, 48, 8, 8]              96
          Conv2d-171              [1, 12, 8, 8]           5,184
      Bottleneck-172             [1, 162, 8, 8]               0
     BatchNorm2d-173             [1, 162, 8, 8]             324
          Conv2d-174              [1, 48, 8, 8]           7,776
     BatchNorm2d-175              [1, 48, 8, 8]              96
          Conv2d-176              [1, 12, 8, 8]           5,184
      Bottleneck-177             [1, 174, 8, 8]               0
     BatchNorm2d-178             [1, 174, 8, 8]             348
          Conv2d-179              [1, 48, 8, 8]           8,352
     BatchNorm2d-180              [1, 48, 8, 8]              96
          Conv2d-181              [1, 12, 8, 8]           5,184
      Bottleneck-182             [1, 186, 8, 8]               0
     BatchNorm2d-183             [1, 186, 8, 8]             372
          Conv2d-184              [1, 48, 8, 8]           8,928
     BatchNorm2d-185              [1, 48, 8, 8]              96
          Conv2d-186              [1, 12, 8, 8]           5,184
      Bottleneck-187             [1, 198, 8, 8]               0
     BatchNorm2d-188             [1, 198, 8, 8]             396
          Conv2d-189              [1, 48, 8, 8]           9,504
     BatchNorm2d-190              [1, 48, 8, 8]              96
          Conv2d-191              [1, 12, 8, 8]           5,184
      Bottleneck-192             [1, 210, 8, 8]               0
     BatchNorm2d-193             [1, 210, 8, 8]             420
          Conv2d-194              [1, 48, 8, 8]          10,080
     BatchNorm2d-195              [1, 48, 8, 8]              96
          Conv2d-196              [1, 12, 8, 8]           5,184
      Bottleneck-197             [1, 222, 8, 8]               0
     BatchNorm2d-198             [1, 222, 8, 8]             444
          Conv2d-199              [1, 48, 8, 8]          10,656
     BatchNorm2d-200              [1, 48, 8, 8]              96
          Conv2d-201              [1, 12, 8, 8]           5,184
      Bottleneck-202             [1, 234, 8, 8]               0
     BatchNorm2d-203             [1, 234, 8, 8]             468
          Conv2d-204              [1, 48, 8, 8]          11,232
     BatchNorm2d-205              [1, 48, 8, 8]              96
          Conv2d-206              [1, 12, 8, 8]           5,184
      Bottleneck-207             [1, 246, 8, 8]               0
     BatchNorm2d-208             [1, 246, 8, 8]             492
          Conv2d-209              [1, 48, 8, 8]          11,808
     BatchNorm2d-210              [1, 48, 8, 8]              96
          Conv2d-211              [1, 12, 8, 8]           5,184
      Bottleneck-212             [1, 258, 8, 8]               0
     BatchNorm2d-213             [1, 258, 8, 8]             516
          Conv2d-214              [1, 48, 8, 8]          12,384
     BatchNorm2d-215              [1, 48, 8, 8]              96
          Conv2d-216              [1, 12, 8, 8]           5,184
      Bottleneck-217             [1, 270, 8, 8]               0
     BatchNorm2d-218             [1, 270, 8, 8]             540
          Conv2d-219              [1, 48, 8, 8]          12,960
     BatchNorm2d-220              [1, 48, 8, 8]              96
          Conv2d-221              [1, 12, 8, 8]           5,184
      Bottleneck-222             [1, 282, 8, 8]               0
     BatchNorm2d-223             [1, 282, 8, 8]             564
          Conv2d-224              [1, 48, 8, 8]          13,536
     BatchNorm2d-225              [1, 48, 8, 8]              96
          Conv2d-226              [1, 12, 8, 8]           5,184
      Bottleneck-227             [1, 294, 8, 8]               0
     BatchNorm2d-228             [1, 294, 8, 8]             588
          Conv2d-229              [1, 48, 8, 8]          14,112
     BatchNorm2d-230              [1, 48, 8, 8]              96
          Conv2d-231              [1, 12, 8, 8]           5,184
      Bottleneck-232             [1, 306, 8, 8]               0
     BatchNorm2d-233             [1, 306, 8, 8]             612
          Conv2d-234              [1, 48, 8, 8]          14,688
     BatchNorm2d-235              [1, 48, 8, 8]              96
          Conv2d-236              [1, 12, 8, 8]           5,184
      Bottleneck-237             [1, 318, 8, 8]               0
     BatchNorm2d-238             [1, 318, 8, 8]             636
          Conv2d-239              [1, 48, 8, 8]          15,264
     BatchNorm2d-240              [1, 48, 8, 8]              96
          Conv2d-241              [1, 12, 8, 8]           5,184
      Bottleneck-242             [1, 330, 8, 8]               0
     BatchNorm2d-243             [1, 330, 8, 8]             660
          Conv2d-244              [1, 48, 8, 8]          15,840
     BatchNorm2d-245              [1, 48, 8, 8]              96
          Conv2d-246              [1, 12, 8, 8]           5,184
      Bottleneck-247             [1, 342, 8, 8]               0
     BatchNorm2d-248             [1, 342, 8, 8]             684
          Linear-249                    [1, 10]           3,430
================================================================
Total params: 769,162
Trainable params: 769,162
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 68.36
Params size (MB): 2.93
Estimated Total Size (MB): 71.31
----------------------------------------------------------------
```

# 3. 复现CIFAR10分类结果



```python
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os
import sys
import math
import shutil
import setproctitle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/densenet.base'
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)

    trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    trainLoader = DataLoader(
        dset.CIFAR10(root='cifar', train=True, download=True,
                     transform=trainTransform),
        batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(
        dset.CIFAR10(root='cifar', train=False, download=True,
                     transform=testTransform),
        batch_size=args.batchSz, shuffle=False, **kwargs)

    net = DenseNet(input_size = 32, growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=10)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        test(args, epoch, net, testLoader, optimizer, testF)
        torch.save(net, os.path.join(args.save, 'latest.pth'))
        os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.item(), err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.item(), err))
        trainF.flush()

def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-1
        elif epoch == 150: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__=='__main__':
    main()
```

Error: 5.98%

<video controls="" autoplay="" name="media"><source src="train.mp4" type="video/mp4"></video>

![](./sgd-loss-error.png)



# 拓展：Densenet + YOLOV4

Densenet作为backbone对图片进行特征提取，再将这些特征当作YOLO的输入。

https://github.com/messileo1/MobileNet_GhostNet_DenseNet_ResNet_Vgg_Yolov4.git

![](./densenet121_yolov4.png)

![](./yoloRes.png)


# Reference

https://github.com/messileo1/MobileNet_GhostNet_DenseNet_ResNet_Vgg_Yolov4.git
https://github.com/bamos/densenet.pytorch.git

# Citations

```
@misc{densenetCookbook,
  title = {{the final presentaion of the Big Data Mining in SUFE}},
  author = {Siying.Xu},
  howpublished = {\url{https://github.com/Alixiying/SUFECourse}},
  note = {Accessed: [Insert date here]}
}
```
