# 命令行大工具

## `tmux`

`tmux`在连接服务器时很有用：当我们要运行一些不终止的程序，但又想关注这个程序输出了什么

- 列出所有会话：`tmux ls`
- 新建会话：`tmux new -s <SESSION_NAME>`
- 连接到会话：`tmux attach -t <SESSION_NAME>`

需要前缀键(默认为`Ctrl+b`)的操作：

- 分离当前会话：前缀键+`d`
- 创建新的窗口：前缀键+`c`
- 进入命令模式：前缀键+`:`

其它：

- 设置tmux窗口内容可以用鼠标滚轮上下滑动
  - 创建/打开`~/.tmux.conf`
  - 写入`set -g mouse on` (对于tmux 2.1及更新版本)
  - 在tmux对话中进入命令模式：前缀键 + `:`
  - 输入`source-file ~/.tmux.conf`，按下`Enter`

## `git`

- 设置用户信息（单个仓库）

  ```
  git config user.name "Your Name"
  git config user.email "your.email@example.com"
  ```

  当多人共享一台服务器时，比较适合这个操作。

- 设置用户信息（全局）

  ```
  git config --global user.name "Your Name"
  git config --global user.email "your.email@example.com"
  ```
