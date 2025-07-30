# 代理与镜像(mirror)设置

## 注意事项
- 普通用户和 `root` 用户的环境变量是分开的。如果在普通用户下设置了代理（例如通过 `http_proxy` 或 `https_proxy` 环境变量），这些设置不会自动应用到 `sudo` 环境中。解决方法有：
  - 显式地将代理设置传递给 `sudo` 环境：`sudo http_proxy=http://your-proxy:port https_proxy=http://your-proxy:port ...`
  - 或者使用 `-E` 参数保留当前用户的环境变量：`sudo -E ...`

## Ubuntu

### apt
先运行`apt update`，成功的话就可以直接用了。如果失败的话，首先我们可以考虑把apt的镜像源换成国内镜像源。这里以Ubuntu 22.04 LTS为例：

- 先备份一份旧的`/etc/apt/sources.list`，可以用`mv`或者`cp`，如果需要放在原`/etc/apt/`目录中则需要管理员权限。

- 修改`/etc/apt/sources.list`，可以直接用下面的脚本（这里选用NJU的镜像源[^2]）：

  ```bash
  echo "deb http://mirror.nju.edu.cn/ubuntu/ jammy main restricted universe multiverse"                > /etc/apt/sources.list 
  echo "deb-src http://mirror.nju.edu.cn/ubuntu/ jammy main restricted universe multiverse"           >> /etc/apt/sources.list 
  echo "deb http://mirror.nju.edu.cn/ubuntu/ jammy-security main restricted universe multiverse"      >> /etc/apt/sources.list 
  echo "deb-src http://mirror.nju.edu.cn/ubuntu/ jammy-security main restricted universe multiverse"  >> /etc/apt/sources.list 
  echo "deb http://mirror.nju.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse"       >> /etc/apt/sources.list 
  echo "deb-src http://mirror.nju.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse"   >> /etc/apt/sources.list 
  echo "deb http://mirror.nju.edu.cn/ubuntu/ jammy-proposed main restricted universe multiverse"      >> /etc/apt/sources.list 
  echo "deb-src http://mirror.nju.edu.cn/ubuntu/ jammy-proposed main restricted universe multiverse"  >> /etc/apt/sources.list 
  echo "deb http://mirror.nju.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse"     >> /etc/apt/sources.list 
  echo "deb-src http://mirror.nju.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list 
  ```

  如果是其它版本的Ubuntu，则参考每个镜像源网站的指导。例如，NJU镜像源对Ubuntu下的镜像指导在此页面：https://mirror.nju.edu.cn/mirrorz-help/ubuntu/?mirror=NJU

如果不想用镜像源，则可以考虑使用代理

### 通过环境变量设置代理
- 在当前shell中设置代理：
  ```bash
  export http_proxy=http://proxy-server:port
  export https_proxy=http://proxy-server:port
  export socks_proxy=socks5://proxy-server:port
  ```
  - 清除当前shell中的代理：
    ```bash
    unset http_proxy
    unset https_proxy
    ```
  
- 为当前用户设置永久代理：把`export`放到`~/.bashrc`中
  ```bash
  echo "export http_proxy=http://proxy-server:port" >> ~/.bashrc
  echo "export https_proxy=http://proxy-server:port" >> ~/.bashrc
  source ~/.bashrc
  ```
  
- 为所有用户设置永久代理：修改`/etc/environment`<br/>`/etc/environment` 是一个系统级的配置文件，所有用户和进程都会继承该文件中定义的环境变量。修改 `/etc/environment` 后，代理设置不会立即生效。你需要重新登录系统或手动执行 `source /etc/environment` 命令来使更改生效。
  ```bash
  http_proxy=http://proxy-server:port
  https_proxy=http://proxy-server:port
  ftp_proxy=http://proxy-server:port
  no_proxy="localhost,127.0.0.1"
  ```



### 常见软件的代理配置

有些软件可能不走环境变量的代理，或者需要单独走代理，此时需要为其单独配置代理。

#### apt

- 在`/etc/apt/apt.conf.d/95proxies`中添加以下内容

  ```bash
  Acquire::http::Proxy "http://proxy-server:port";
  Acquire::https::Proxy "http://proxy-server:port";
  ```

#### Git

- 使用以下命令

  ```bash
  git config --global http.proxy http://proxy-server:port
  git config --global https.proxy http://proxy-server:port
  ```

  或者修改配置文件：

  - 全局配置文件：`~/.gitconfig`
  - 当前仓库配置文件：`.git/config`

  ```
  [http]
         proxy = http://127.0.0.1:7890
  ```

#### wget

- 临时使用

  ```bash
  wget -e use_proxy=yes -e http_proxy=http://proxy-server:port -e https_proxy=http://proxy-server:port <target_url>
  ```

- 配置：`~/.wgetrc`

  ```bash
  echo "use_proxy=yes" >> ~/.wgetrc
  echo "http_proxy=http://proxy-server:port" >> ~/.wgetrc
  echo "https_proxy=http://proxy-server:port" >> ~/.wgetrc
  ```

#### curl

- 在命令中使用`curl -x http://proxy-server:port http://example.com`

#### ssh

- 通过`~/.ssh/config`配置

  ```
  Host <目标主机名>
      ProxyCommand nc -X 5 -x <proxy-server>:<port> %h %p
  ```

### 其它

- Python程序下载数据集：可以直接在代码中设置环境变量

  ```python
  import os
  os.environ['HTTP_PROXY'] = 'http://your_proxy_address:port'
  os.environ['HTTPS_PROXY'] = 'http://your_proxy_address:port'
  ```

  

## WSL2

WSL2中，如果是NAT模式，则无法使用Windows上配置的代理，而且此时Windows主机不是通过127.0.0.1访问。

- 首先我们需要获取Windows的IP：
  ```bash
  ip route show | grep -i default | awk '{ print $3}'
  ```
- 用Windows的IP设置代理：
  ```bash
  export http_proxy=http://<windows_ip>:<proxy_port>
  export https_proxy=http://<windows_ip>:<proxy_port>
  ```



## docker
### docker daemon

Daemon的代理配置见[Daemon proxy configuration](https://docs.docker.com/engine/daemon/proxy/#daemon-configuration)。官方提供了两种方法，这里仅介绍其中的一种方法：

- 在`/etc/docker/daemon.json`中设置

  ```json
  {
    "proxies": {
      "http-proxy": "http://proxy.example.com:3128",
      "https-proxy": "https://proxy.example.com:3129",
      "no-proxy": "*.test.example.com,.example.org,127.0.0.0/8"
    }
  }
  ```

  - 重启docker daemon：`sudo systemctl restart docker`

### docker client[^1]

在`~/.docker/config.json`添加以下内容：

```json
{
 "proxies": {
   "default": {
     "httpProxy": "http://proxy.example.com:3128",
     "httpsProxy": "https://proxy.example.com:3129",
     "noProxy": "*.test.example.com,.example.org,127.0.0.0/8"
   }
 }
}
```

---
[^1]: Use a proxy server with the Docker CLI: https://docs.docker.com/engine/cli/proxy/

[^2]: http://mirror.nju.edu.cn/
