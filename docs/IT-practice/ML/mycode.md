# 代码整理

## Import

```python
from configuration import *
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from tqdm import tqdm  # 用于显示漂亮的进度条
from typing import Optional, List, Union, Any, Tuple, Callable
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 必须使用非交互式后端
import matplotlib.pyplot as plt
```

## 模型

### 返回中间结果的分类器

```python
class Classifier(nn.Module, ABC):
    """
    一个抽象的分类器基类 (Abstract Base Class)。

    它定义了一个标准接口，要求所有子类实现：
    1. 一个 'fc_layer' 属性，用于返回模型的最后一个全连接(分类)层。
    2. 一个 'forward' 方法，该方法必须能够根据参数选择性地返回
       中间特征或特征列表。
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def fc_layer(self) -> nn.Linear:
        """
        抽象属性。
        子类必须覆盖此属性，返回其最后一个全连接层 (nn.Linear)。
        
        示例：
            @property
            def fc_layer(self):
                return self.my_classification_head
        """
        raise NotImplementedError("子类必须实现 'fc_layer' 属性")

    def get_fc_layer_copy(self) -> nn.Linear:
        """
        获取最后一个全连接层 (fc_layer) 的一个深拷贝 (Deep Copy)。
        
        这对于在不修改原始模型的情况下，
        重用或修改分类头非常有用 (例如，在知识蒸馏中)。
        
        返回:
            nn.Linear: self.fc_layer 的一个深拷贝。
        """
        # 使用 copy.deepcopy 来确保权重和计算图被完全复制
        return copy.deepcopy(self.fc_layer)

    @abstractmethod
    def forward(self,
                x: torch.Tensor,
                return_feature: bool = False,
                return_feature_list: bool = False
               ) -> Union[torch.Tensor, 
                          Tuple[torch.Tensor, torch.Tensor], 
                          Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        抽象的前向传播方法。

        子类必须实现此方法以处理三种返回情况：

        1. (默认) return_feature=False, return_feature_list=False:
           只返回分类的 logits (torch.Tensor)。

        2. return_feature=True:
           返回 (logits, final_feature) (Tuple[torch.Tensor, torch.Tensor])。
           'final_feature' 通常是 fc_layer 之前的特征向量。

        3. return_feature_list=True:
           返回 (logits, feature_list) (Tuple[torch.Tensor, List[torch.Tensor]])。
           'feature_list' 包含了一个或多个中间层的特征图/向量。
           
        注意：如果 'return_feature_list' 为 True，它应优先于 'return_feature'。
        """
        raise NotImplementedError("子类必须实现 'forward' 方法")
```

## 数据集

### (路径, 标签)类数据集

```python
class TxtFileDataset(Dataset):
    """
    一个用于从 .txt 文件加载图像路径和标签的自定义数据集。
    .txt 文件的格式应为:
    relative/path/to/image.png 0
    relative/path/to/another.jpg 1
    ...
    """
    
    def __init__(self, path_file: Path, data_dir: Path, transform=None):
        """
        参数:
        path_file (Path): 指向 .txt 文件的路径。
        data_dir (Path): .txt 文件中相对路径的根目录。
        transform (callable, optional): 应用于样本的 transform。
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        print(f"Loading data index from: {path_file}")
        print(f"Using data root directory: {data_dir}")

        # 打开 .txt 文件并解析
        with open(path_file, 'r') as f:
            for line in f:
                line = line.strip()
                # 跳过空行
                if not line:
                    continue
                
                parts = line.split()
                
                if len(parts) == 2:
                    relative_path = parts[0]
                    label = int(parts[1])
                    
                    # 组合根目录和相对路径
                    full_path = data_dir / relative_path
                    
                    self.image_paths.append(full_path)
                    self.labels.append(label)
                else:
                    print(f"Warning: Skipping malformed line: '{line}'")

        print(f"Found {len(self.image_paths)} samples.")

    def __len__(self) -> int:
        """返回数据集中的样本总数"""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        根据索引 idx 获取一个样本。
        """
        # 获取图像路径和标签
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        try:
            # 使用 PIL (Pillow) 加载图像
            # .convert("RGB") 确保图像是 3 通道的，防止单通道灰度图导致错误
            image = Image.open(img_path).convert("RGB")
            
        except FileNotFoundError:
            print(f"Error: File not found at {img_path}")
            # 返回一个虚拟的黑色图像和标签
            image = Image.new("RGB", (32, 32), (0, 0, 0))
            # 或者您可以选择抛出异常
            # raise FileNotFoundError(f"Image not found: {img_path}")
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (32, 32), (0, 0, 0))

        # 应用 transform (例如 ToTensor(), Normalize())
        if self.transform:
            image = self.transform(image)

        return image, label
```

### 分离标签

```python
def get_labels(dataset: Dataset, 
                 batch_size: int = 256, 
                 num_workers: int = 0) -> torch.Tensor:
    """
    从 Pytorch 数据集中提取所有标签并返回一个 Tensor。

    此函数会智能检测数据集类型：
    1. 如果是 TensorDataset，它会直接访问 .tensors[1]，速度极快。
    2. 如果是其他 Dataset (如 ImageFolder)，它会使用 DataLoader 迭代。

    Args:
        dataset (Dataset): 原始数据集 (如 CIFAR10) 或
                           特征数据集 (TensorDataset)。
        batch_size (int, optional): 仅在需要迭代时使用 (非 TensorDataset).
        num_workers (int, optional): 仅在需要迭代时使用.

    Returns:
        torch.Tensor: 包含所有标签的一维张量 (在 CPU 上)。
    """
    
    # --- 1. 最高效的情况: TensorDataset ---
    # (这包括我们的 'feature_dataset' 和
    # 任何已包装为 TensorDataset 的 'original_dataset')
    if isinstance(dataset, TensorDataset):
        print("检测到 TensorDataset，正在直接提取标签...")
        
        # 假设: (data, label)
        # dataset.tensors 是一个 (tensor_of_data, tensor_of_labels) 元组
        if len(dataset.tensors) >= 2:
            # .tensors[1] 就是我们需要的标签张量
            labels_tensor = dataset.tensors[1]
            # 确保它在 CPU 上
            return labels_tensor.cpu()
        else:
            raise ValueError("输入的 TensorDataset 格式不正确，应至少包含两个张量 (data, labels)。")
    
    # --- 2. 通用情况: 任何其他 Dataset (如 ImageFolder, CIFAR10) ---
    # 我们必须迭代。使用 DataLoader 是最高效的方式。
    print(f"检测到通用 Dataset，正在使用 DataLoader (batch_size={batch_size}) 迭代提取标签...")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 必须为 False 以保持顺序
        num_workers=num_workers,
        pin_memory=False
    )
    
    all_labels = []
    
    try:
        # 假设 DataLoader 产生 (data, labels)
        for _, labels in tqdm(dataloader, desc="提取标签进度"):
            all_labels.append(labels)
            
    except Exception as e:
        print(f"\n使用 DataLoader 批量提取时出错: {e}")
        print("这可能是因为数据集的 __getitem__ 不返回 (data, label) 元组。")
        print("正在尝试逐个样本回退 (这可能会很慢)...")
        
        # 回退(Fallback)方法：逐个迭代 (非常慢)
        all_labels = []
        for i in tqdm(range(len(dataset)), desc="回退 (逐个)"):
            try:
                # 尝试解包，假设返回 (data, label)
                _, label = dataset[i]
                # 确保是标量张量
                if not isinstance(label, torch.Tensor):
                    label = torch.tensor(label)
                all_labels.append(label)
            except Exception as inner_e:
                print(f"无法处理索引 {i}: {inner_e}")
                print("请检查您的 Dataset 实现。")
                return None # 失败

        if not all_labels:
            return torch.empty(0)
        
        # 手动堆叠 (因为它们是单个样本)
        return torch.stack(all_labels, dim=0).cpu()

    # --- 3. 合并 DataLoader 的结果 ---
    if not all_labels:
        print("数据集中未找到标签。")
        return torch.empty(0)
        
    try:
        # 合并所有批次
        final_labels = torch.cat(all_labels, dim=0)
        return final_labels.cpu()
    except RuntimeError as e:
        print(f"合并标签时出错: {e}")
        print("请检查数据集中所有标签是否具有相同的类型/形状。")
        return None
```



## 分离正确分类和错误分类的样本

```python
def get_classified_indices(model: nn.Module,
                           dataset: Dataset,
                           batch_size: int = 128,
                           num_workers: int = 4,
                           device: Optional[str] = None) -> Tuple[List[int], List[int]]:
    """
    遍历数据集，返回所有被模型 *正确分类* 和 *误分类* 的样本的索引。

    Args:
        model (nn.Module): 用于评估的模型。
        dataset (Dataset): 要检查的数据集。
        batch_size (int, optional): 批处理大小. Defaults to 128.
        num_workers (int, optional): DataLoader 的工作进程数. Defaults to 4.
        device (str, optional): 'cuda' or 'cpu'. 
                                如果为 None, 自动检测. Defaults to None.

    Returns:
        Tuple[List[int], List[int]]: 
            - correctly_classified_indices (List[int]): 正确分类的索引。
            - misclassified_indices (List[int]): 误分类的索引。
    """

    # 1. 自动检测设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. 准备模型
    model = model.to(device)
    model.eval()  # !! 必须设置为评估模式

    # 3. 准备 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # !! 必须为 False 来保证索引正确
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )

    # 4. 存储结果的列表
    all_misclassified_indices = []
    all_correctly_classified_indices = []
    
    print(f"--- 正在 {device} 上检查分类样本 ---")
    total_samples = 0

    # 5. 迭代和检查
    with torch.no_grad():  # !! 必须禁用梯度计算
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="检查进度")):
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            total_samples += labels.size(0) # 累加样本总数
            
            logits = model(inputs)
            predictions = torch.argmax(logits, dim=1)
            
            # --- 核心逻辑：计算两种掩码 ---
            
            # (N,) 形状的布尔张量 (True 代表正确分类)
            correct_mask = (predictions == labels)
            
            # (N,) 形状的布尔张量 (True 代表误分类)
            misclassified_mask = (predictions != labels)
            
            # --- 分别处理两种索引 ---

            # .where() 返回一个元组，我们取第一个元素
            local_correct_indices = torch.where(correct_mask)[0]
            local_misclassified_indices = torch.where(misclassified_mask)[0]
            
            # 计算全局索引的起始点
            batch_start_index = batch_idx * batch_size
            
            # --- 存储正确分类的索引 ---
            if local_correct_indices.numel() > 0:
                global_correct_indices = local_correct_indices + batch_start_index
                all_correctly_classified_indices.extend(
                    global_correct_indices.cpu().numpy().tolist()
                )
            
            # --- 存储误分类的索引 ---
            if local_misclassified_indices.numel() > 0:
                global_misclassified_indices = local_misclassified_indices + batch_start_index
                all_misclassified_indices.extend(
                    global_misclassified_indices.cpu().numpy().tolist()
                )

    print(f"检查完毕。")
    print(f"  总样本数: {total_samples} (应与 len(dataset) 一致)")
    print(f"  正确分类: {len(all_correctly_classified_indices)}")
    print(f"  错误分类: {len(all_misclassified_indices)}")
    
    return all_correctly_classified_indices, all_misclassified_indices
```

