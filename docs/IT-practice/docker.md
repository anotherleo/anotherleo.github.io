# Docker

- 查看已有镜像：`docker images`或`docker image ls`
- 查看已有容器：`docker ps -a`

## 创建容器

- 普通创建

  ```bash
  docker run -d \
    --name my-app \
    -p 8080:80 \
    my-image:latest
  ```

   `-d` (detached) 标志告诉 Docker 在后台运行容器，并打印出容器 ID。容器启动后，命令行会立即返回到你的终端，但容器在后台保持运行。

  进入容器：`docker attach`或者`docker exec -it`。

  退出容器：`Ctrl + P` 后跟 `Ctrl + Q`

  启动容器：`docker start <容器名称或ID>`

### 使用`docker-compose.yml`创建容器

```yaml
services:
  my_web_app: # 服务名称，可以自定义
    image: my-app:latest # 指定你要使用的镜像
    container_name: my-app-instance # 指定容器名称
    
    ports:
      - "8080:80" # 端口映射：本地8080端口映射到容器80端口
      
    volumes:
      # 目录映射列表
      - ./config:/etc/app/config # 映射 1: 将本地当前目录下的 'config' 文件夹映射到容器的 /etc/app/config
      - ./data:/var/lib/app/data # 映射 2: 将本地当前目录下的 'data' 文件夹映射到容器的 /var/lib/app/data
      # 注意：如果使用相对路径（如 ./config），它指的是 docker-compose.yml 文件所在的目录
```

在`docker-compose.yml`所在目录下用`docker compose up -d`创建容器。
