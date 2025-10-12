# Linux机器管理

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

### 账户的停用与清理

- 锁定账户：`sudo passwd -l [username]`
- 备份数据：`sudo tar -czf /archives/[username]_backup.tar.gz /home/[username]`
- 删除账户：`sudo userdel -r [username]` 
  - `-r`会一并删除家目录

### 查看用户使用

- `last`：查看所有用户、终端、时间和 IP 地址的登录和注销历史（`/var/log/wtmp`）
- `who`：查看当前在线的用户
- `history`：查看当前用户执行过的命令历史<br/>如果要查看其它用户的执行历史，则需要在root权限去查看：`sudo cat /home/otheruser/.bash_history`

## 资源管理

### 日常资源监控

- CPU和内存：`htop`或者`top`
- 磁盘空间：`df -h`
  - `sudo du -sh /home/*`：查看每个用户占用了多少空间
- GPU：`nvidia-smi`
- 资源限制：
  - 磁盘配额：配置 `quota`
  - 进程限制：配置 `/etc/security/limits.conf` 文件
- 实施备份：定期用`sudo rsync`或者`tar`备份

## 其它

- 系统更新：`sudo apt update` 和 `sudo apt upgrade -y`
- 日志审查：
  - `sudo grep "Failed password" /var/log/auth.log` 查看失败的登录尝试，判断是否有暴力破解行为。
  - `sudo journalctl -f` 实时查看系统日志
