# SSH

- 查看本地公钥

  ```bash
  ls ~/.ssh/
  ```

  公钥名通常为`id_rsa.pub`

- 将公钥上传至服务器

  - 使用ssh-copy-id命令复制公钥

    ```bash
    ssh-copy-id -i ~/.ssh/id_rsa.pub username@remote_host
    ```

    
