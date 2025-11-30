# PyTorch

```python
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
```



## 数据结构

### 基本数据结构

- `torch.Tensor`

### `torchvision.transforms`

- `transforms.Compose(transforms)`：把`transforms`中的多个transform复合成一个
- `transforms.ToTensor()`：(1)将 PIL 图像或 `numpy.ndarray` 转换为 `torch.Tensor`; (2)将像素值从 `[0, 255]` 缩放到 `[0.0, 1.0]`
- `transforms.ToPILImage()`：将 Tensor 转换回 PIL 图像
- `transforms.Normalize(mean, std)`

### 数据集

- `torch.utils.data.Dataset`
- `torchvision.datasets.DatasetFolder` -> `torchvision.datasets.VisionDataset` -> `torch.utils.data.Dataset`
  适用于加载以下这类数据
  ```
  root/
  ├── class_x
  │   ├── xxx.ext
  │   ├── xxy.ext
  │   └── ...
  └── class_y
      ├── 123.ext
      ├── abc.ext
      └── ...
  ```
- `torchvision.datasets.ImageFolder` -> `torchvision.datasets.DatasetFolder`
  - `root:Union[str, Path]`
  - `transform=None`
- `torch.utils.data.DataLoader`

  - `dataset`
  - `batch_size=1`
  - `shuffle=None`

- `torch.utils.data.Subset`

  - `dataset`
  - `indices`

  

## 方法

## Torchvision数据集

### MNIST

### CIFAR-10

### ImageNet
