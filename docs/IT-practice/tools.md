# 常用命令行工具

## `tmux`

`tmux`在连接服务器时很有用：当我们要运行一些不终止的程序，但又想关注这个程序输出了什么。

- 列出所有会话：`tmux ls`
- 新建会话：`tmux new -s <SESSION_NAME>`
- 连接到会话：`tmux attach -t <SESSION_NAME>`

需要前缀键(默认为`Ctrl+b`)的操作：

- 分离当前会话：前缀键+`d`
- 创建新的窗口：前缀键+`c`