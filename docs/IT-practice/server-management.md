# Linux服务器管理

修改密码：

```bash
$ passwd
Changing password for your_username.
(current) UNIX password: 
New password: 
Retype new password: 
passwd: password updated successfully
```

## 了解服务器基本信息

- 操作系统版本：`lsb_release -a` 或者 `cat /etc/redhat-release`
- 磁盘分区：`df -h`
- CPU：`lscpu`
- 内存：`free -m`
- 现有用户：`cat /etc/passwd` （或者更加精确的 `cat /etc/passwd | grep "/bin/bash"` ）

## 用户管理

### 创建新账号

- 创建用户：`sudo useradd -m -s /bin/bash [username]`
  - `-m` 会自动创建用户的家目录 (`/home/[username]`)。
  - `-s /bin/bash` 为用户指定默认的shell。
- 设置初始密码：`sudo passwd [username]`
- 强制用户下次登录时修改密码：`sudo chage -d 0 [username]`

自动化脚本：

```bash
#!/bin/bash
# create_users.sh

# 检查权限
if [ "$EUID" -ne 0 ]; then
  echo "请使用 sudo 或以 root 权限运行。"
  exit 1
fi

# 初始设置
DEFAULT_PASS="123456"

# 处理用户列表的函数
process_user() {
    local username=$1
    if id "$username" &>/dev/null; then
        echo "[跳过] 用户 '$username' 已存在。"
    else
        # 创建用户
        useradd -m "$username"
        # 设置初始密码
        echo "$username:$DEFAULT_PASS" | chpasswd
        # 强制下次登录修改密码
        passwd -e "$username"
        echo "[成功] 用户 '$username' 已创建。"
    fi
}

# 判断输入是文件还是参数列表
if [ -f "$1" ]; then
    # 模式 A: 从文件读取
    echo "正在从文件 $1 中读取用户..."
    while IFS= read -r line || [ -n "$line" ]; do
        # 过滤空行和注释
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        process_user "$line"
    done < "$1"
else
    # 模式 B: 从命令行参数读取
    if [ $# -eq 0 ]; then
        echo "用法:"
        echo "  方式 1 (参数): $0 user1 user2 user3"
        echo "  方式 2 (文件): $0 user_list.txt"
        exit 1
    fi
    for user in "$@"; do
        process_user "$user"
    done
fi

echo "--- 任务完成 ---"
```

### 账户的停用与清理

- 锁定账户：`sudo passwd -l [username]`
- 备份数据：`sudo tar -czf /archives/[username]_backup.tar.gz /home/[username]`
- 删除账户：`sudo userdel -r [username]` 
  - `-r`会一并删除家目录

### 查看用户使用

- `last`：查看所有用户、终端、时间和 IP 地址的登录和注销历史（`/var/log/wtmp`）
- `who`：查看当前在线的用户
- `history`：查看当前用户执行过的命令历史<br/>如果要查看其它用户的执行历史，则需要在root权限去查看：`sudo cat /home/otheruser/.bash_history`

### 数据管理

在数据盘`/data`下为每个用户创建一个自己的目录：

1. 配置父目录

   ```bash
   chown root:root /data
   chmod 755 /data
   ```

2. 用脚本分配

   ```bash
   #!/bin/bash
   
   # 检查是否提供了参数
   if [ $# -eq 0 ]; then
       echo "用法: $0 user1 user2 user3 ..."
       exit 1
   fi
   
   # 使用 "$@" 获取命令行输入的所有用户名
   for user in "$@"; do
       dir="/data/$user"
       
       # 首先检查用户是否存在，否则 chown 会报错
       if ! id "$user" &>/dev/null; then
           echo "错误: 用户 $user 不存在，跳过创建目录。"
           continue
       fi
   
       if [ ! -d "$dir" ]; then
           mkdir -p "$dir"
           chown "$user":"$user" "$dir"
           chmod 700 "$dir"
           echo "已为用户 $user 创建目录: $dir"
       else
           echo "目录 $dir 已存在，跳过。"
       fi
   done
   ```

## 资源管理

### 日常资源监控

- CPU和内存：`htop`或者`top`
- 磁盘空间：`df -h`
  - `sudo du -sh /home/*`：查看每个用户占用了多少空间
  - `sudo du -h --max-depth=1 /path/to/... | sort -hr | head -n 10`：查看目标路径下占用空间最大的十个目录/文件
- GPU：`nvidia-smi`
- 资源限制：
  - 磁盘配额：配置 `quota`
  - 进程限制：配置 `/etc/security/limits.conf` 文件
- 实施备份：定期用`sudo rsync`或者`tar`备份

## 软件安装

### 用户组

一个常见的做法是把共享第三方软件安装在`/opt`目录下

- 创建一个专门的组：`sudo groupadd software`

- 修改 `/opt` 的所属组：`sudo chgrp -R software /opt`

- 设置权限

  ```shell
  # 赋予组员写入权限，其他人只能读取和执行
  sudo chmod -R 775 /opt
  # 设置 GID 位，使得以后在 /opt 下创建的新文件自动继承 software 组
  sudo chmod g+s /opt
  ```

- 将需要“安装/管理”权限的用户加入该组：`sudo usermod -aG software 用户名`

## 其它

- 系统更新：`sudo apt update` 和 `sudo apt upgrade -y`
- 日志审查：
  - `sudo grep "Failed password" /var/log/auth.log` 查看失败的登录尝试，判断是否有暴力破解行为。
  - `sudo journalctl -f` 实时查看系统日志
