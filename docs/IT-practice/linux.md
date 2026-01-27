# Linux机制

## 权限表示法

Linux 权限分为三类对象：Owner (u)、Group (g) 和 Others (o)；每类对象有三种权限：Read (r)、Write (w) 和 Execute (x)。对于目录，`rwx` 的含义与普通文件略有不同：

- r (Read)：可以列出目录里的文件名（`ls`）。
- w (Write)：可以在目录内创建、删除、重命名文件。
- x (Execute)：允许“进入”目录（`cd`）或访问其内部的文件。注意：如果一个目录没有 `x` 权限，即使你有 `r` 权限能看到文件名，也无法查看文件内容或属性。

权限的两种表示法：

- 数字表示法：
  - 4：读取 (r)，2：写入 (w)，1：执行 (x)，0：无权限 (---)
  - 777：所有人拥有所有权限（极不安全，慎用）。
  - 755：所有者全开，组和其他人可读、可执行（常用于脚本和目录）。
  - 644：所有者可读写，组和其他人只读（常用于普通文档）。
  - 600：只有所有者可读写，其他人完全无法访问（常用于密钥文件）。
- 字母表示法（相对权限）
  - 用户类型：`u` (Owner), `g` (Group), `o` (Others), `a` (All, 所有人)
  - 操作符：`+` (添加), `-` (移除), `=` (精确设置)
  - 权限：`r`, `w`, `x`
  - `chmod +x script.sh`：给所有人添加执行权限。
  - `chmod g-w file.txt`：移除组用户的写权限。
  - `chmod u=rwx,g=rx,o=r file.txt`：精确分配不同权限。

## 用户组

查看组信息

- `groups [<user>]`: 查看用户所属的组。
- `id [<user>]`: 查看用户的详细身份信息（包括 UID、GID 和所属的所有组 ID）。
- `getent group`: 列出系统中的所有组（数据来自 `/etc/group`）。
  - 查看特定组：`getent group <group>`

创建与删除组：

- `groupadd`: 创建一个新组。
- `groupdel`: 删除一个组。
- `groupmod`: 修改组名或 GID。

用户与组的关联（成员管理）

- `usermod`: 修改用户所属的组。
  - `sudo usermod -aG <组名> <用户名>`: 修改用户所属的组。`-a` 是追加（append），`-G` 是附加组。如果漏写 `-a`，该用户会被从其他原有的附加组中移除，非常危险！
- `gpasswd`: 专门用于管理组成员的工具。
  - `sudo gpasswd -a <用户名> <组名>`: 添加用户到组。
  - `sudo gpasswd -d <用户名> <组名>`: 从组中移除用户。

组与文件权限：

- `chgrp`: 改变文件的所属组。
  - `sudo chgrp dev-team project.txt`
- `chown`: 同时改变所有者和所属组（最常用）。
  - `sudo chown jack:dev-team project.txt` (格式为 `所有者:组`)
- `chmod`: 修改组权限位。
  - `sudo chmod <权限> <文件>`
  - `-R`: 递归修改
    - `sudo chmod -R 755 /var/www/html`