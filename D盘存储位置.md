# D 盘存储位置

项目实际存放在 `D:\CodexProjects\board-misuse`。

- `outputs`：数据集、已训练模型、报告、预测图和脚本。
- `work`：独立 Python 训练环境、原始下载文件等。

原 C 盘工作目录中的 outputs 和 work 已改为指向 D 盘的目录联接，不再存放一份项目数据。之前聊天中的文件链接仍然有效；通过原路径写入的内容实际也存放在 D 盘。

数据集 YAML 和图片列表已更新为 D 盘路径。历史训练日志仍保留当时的原始路径。

在 PowerShell 中使用：

```powershell
Set-Location D:\CodexProjects\board-misuse\outputs
..\work\board-env\Scripts\python.exe .\predict_board.py "D:\你的文件夹\厨房视频.mp4"
```

Python 基础安装仍在 C 盘，D 盘的独立环境依赖该基础安装。系统级 pip 下载缓存不属于本项目，此次未迁移。
