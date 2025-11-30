# conda

## 安装conda

每个用户激活conda：

- 首先执行：`eval "$(/opt/anaconda3/bin/conda shell.YOUR_SHELL_NAME hook)"`

  - 这里的`YOUR_SHELL_NAME`要替换成具体的shell名称，例如`bash`或者`zsh`

  - ```bash
    eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
    ```

- 然后执行：`conda init`

## 环境

- 列出conda的所有环境：`conda env list` or `conda info --envs`
- 查看当前的环境：`(my_env_name) user@machine:~$`
- 切换环境：`conda activate my_analysis_env`
- 返回base环境：`conda deactivate`
- 新建环境：`conda create -n (my_env_name) python=3.9`
- 删除环境：`conda env remove --name (my_env_name) ` or `conda remove --name (my_env_name) --all`
- 导出环境：`conda env export > environment.yml`

## 镜像配置(Channels)

Conda官方文档：[Channels](https://docs.conda.io/projects/conda/en/stable/user-guide/concepts/channels.html), [Mirroring channels](https://docs.conda.io/projects/conda/en/stable/user-guide/configuration/mirroring.html)

- 查看镜像源列表：`conda config --show channels`

- 修改`.condarc`来设置：

  ```
  default_channels:
    - https://mirror.nju.edu.cn/anaconda/pkgs/main
    - https://mirror.nju.edu.cn/anaconda/pkgs/r
    - https://mirror.nju.edu.cn/anaconda/pkgs/msys2
  
  custom_channels:
    conda-forge: https://mirror.nju.edu.cn/anaconda/cloud
    pytorch: https://mirror.nju.edu.cn/anaconda/cloud
  ```

  注意这里是通过改default_channels而不是直接把这些加到channels中，channels中始终会有default_channels，我们要把原先default_channels的conda源换成自己的源。

- 设置搜索包时显示 channel URL：`conda config --set show_channel_urls yes`

- 删除镜像源：`conda config --remove channels <您要移除的镜像源地址>`

- 恢复默认镜像源：`conda config --remove-key channels`

## 代理

conda默认使用系统代理，也可以单独设置代理（`~/.condarc`配置的优先级最高）：

- 命令行设置

  ```bash
  conda config --set proxy_servers.http http://<proxy_address>:<port>
  conda config --set proxy_servers.https https://<proxy_address>:<port>
  ```

- 在`~/.condarc`文件中设置

  ```
  proxy_servers:
    http: http://<user>:<pass>@<proxy_address>:<port>
    https: https://<user>:<pass>@<proxy_address>:<port>
  ```


## 环境配置

要点：能用conda先装好的就用conda先装，不要一上来就用pip。

### PyTorch

为了避免装错成CPU版本，对于一个新环境，先使用以下命令安装最基础的包：

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

为了验证是否安装成功，用以下命令：

```
python -c "import torch; print(f'CUDA OK: {torch.cuda.is_available()}')"
```

### 其它机器学习包

```
conda install numpy scipy pandas scikit-learn pillow matplotlib
```

## 其它

### 缓存污染问题

假如缓存的包中出了问题，导致后续所有环境在安装时都出来问题，这时需要把所有包都清理掉：

```
conda clean --all
```

### 更新conda

如果conda安装的地方需要管理员权限，那么需要使用sudo。而使用sudo则可能会遇到两个问题：

1. 找不到conda（因为conda是在用户的`.bashrc`中加载的）
2. 不能连接代理（因为sudo不会继承用户的环境变量）

(1)的解决可以通过用conda的绝对路径，(2)的解决可以用`sudo -E`。