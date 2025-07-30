# 包管理

## conda

- 查找库：`conda search 库名`

- 安装：`conda install 库名`

  - conda找不到时再用pip：`pip install 库名`

- 指定安装源：

  ```bash
  # 使用清华 conda 镜像
  conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ 库名
  
  # 或 pip 镜像
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 库名
  ```

- 查看已安装库：`conda list` 或 `pip list`

- 升级库：`conda update 库名` 或 `pip install --upgrade 库名`

- 卸载库：`conda remove 库名` 或 `pip uninstall 库名`