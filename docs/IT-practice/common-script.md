# 常用脚本

## 处理Markdown文件

### 换行

1. 把多于3行的空白行替换成两行的空白行

   ```
   ```

   

### 脚注

Markdown文件中，脚注的处理是一件比较麻烦的事情。这个时候，如果能用脚本来完成相应的工作，则能大大提高工作效率。此处对可能的函数做拆解：

1. 分离正文和脚注的注释
2. `def getExistingFootnote(content:str) -> {str:str}`
3. `def remove`



### 高亮(`==.*==`)转脚注：`highlight2footnote.py`

脚注由两部分组成，第一部分是编号（`[^<number>]`），第二部分是文档末尾的注释（`[^<number>]:...`）。为了添加一个脚注，同时标记两部分是比较麻烦的。为了方便标注，我这边先用高亮（Mac上typora的快捷键`Shift`+`Command`+`H`）标记为要脚注的地方，然后再用`highlight2footnote.py`一次性对脚注进行处理：



