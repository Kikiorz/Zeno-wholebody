# Hugging Face 数据上传和下载指南

本文档说明如何将 point1~point10 数据上传到 Hugging Face，以及合作者如何下载数据。

## 📋 目录

1. [准备工作](#准备工作)
2. [上传数据](#上传数据)
3. [下载数据](#下载数据)
4. [常见问题](#常见问题)

---

## 准备工作

### 创建 Hugging Face Access Token

1. 登录 https://huggingface.co
2. 访问 https://huggingface.co/settings/tokens
3. 点击 "New token" 创建新token
4. Token设置：
   - **Name**: 随意命名（如 "pick-dataset-upload"）
   - **Type**: 选择 "Write"（写入权限）
5. 点击 "Generate token"
6. **重要**: 立即复制并保存这个token（格式：`hf_xxxxxxxxxxxxx`），关闭页面后无法再查看

---

## 上传数据

### 步骤1：运行上传脚本

```bash
cd /home/zeno-yanan/ZENO-ROS/piper_ros/data_collect/pick

python3 upload_pick_data.py \
  --repo_id your-username/pick-dataset \
  --token hf_your_token_here
```

**参数说明：**
- `your-username`: 替换为你的 Hugging Face 用户名
- `pick-dataset`: 数据集名称（可以自定义）
- `hf_your_token_here`: 替换为你创建的token

**示例：**
```bash
python3 upload_pick_data.py \
  --repo_id zeno-yanan/piper-pick-data \
  --token hf_your_token_here
```

### 步骤2：等待上传完成

- 上传约6.8GB数据需要30分钟到2小时（取决于网络速度）
- 脚本会显示上传进度
- 不要中断上传过程
- 如果上传失败，重新运行命令会自动续传

### 步骤3：设置数据集为公开

1. 访问数据集页面：`https://huggingface.co/datasets/your-username/pick-dataset`
2. 点击 "Settings" 标签
3. 在 "Visibility" 部分，选择 "Public"（公开）
4. 保存设置

### 步骤4：创建README文档

在数据集页面：
1. 点击 "Files and versions"
2. 点击 "Add file" > "Create a new file"
3. 文件名：`README.md`
4. 内容：

```markdown
# Piper Robot Pick Dataset

## 数据集描述
这是用于Piper机器人抓取任务的ROS bag数据集，包含10个不同抓取点的数据。

## 数据结构
- 10个文件夹：point1 ~ point10
- 每个文件夹包含约15个.bag文件
- 总大小：约6.8GB

## 下载方法

### 方法1：使用Python脚本（推荐）
```bash
pip install huggingface_hub
python3 download_pick_data.py --repo_id your-username/pick-dataset
```

### 方法2：使用命令行
```bash
pip install huggingface_hub
huggingface-cli download your-username/pick-dataset --repo-type dataset --local-dir ./pick_data
```

### 方法3：使用Git LFS
```bash
git lfs install
git clone https://huggingface.co/datasets/your-username/pick-dataset
```

## 数据采集信息
- 机器人：Piper双臂机器人
- 传感器：双目鱼眼相机
- 任务类型：物体抓取
```

---

## 下载数据

### 方法1：使用Python脚本（推荐，最简单）

```bash
# 安装依赖
pip install huggingface_hub

# 下载数据
python3 download_pick_data.py --repo_id your-username/pick-dataset
```

数据会下载到 `./pick_data` 目录。

### 方法2：使用命令行工具

```bash
# 安装工具
pip install huggingface_hub

# 下载数据集
huggingface-cli download your-username/pick-dataset \
  --repo-type dataset \
  --local-dir ./pick_data
```

### 方法3：使用Git LFS

```bash
# 安装Git LFS
git lfs install

# 克隆数据集
git clone https://huggingface.co/datasets/your-username/pick-dataset
```

### 方法4：浏览器直接下载

1. 访问数据集页面
2. 点击 "Files and versions"
3. 点击单个文件右侧的下载按钮

---

## 常见问题

### Q: 上传失败怎么办？
**A:** 重新运行上传命令，huggingface_hub会自动续传已上传的部分。

### Q: 可以只上传部分文件夹吗？
**A:** 可以，修改脚本中的 `allow_patterns` 参数。但建议一次性上传完整数据集。

### Q: 合作者下载很慢怎么办？
**A:** 使用方法1的Python脚本，它支持断点续传。或者使用Git LFS的partial clone功能。

### Q: 如何删除或更新数据集？
**A:** 在Hugging Face网站上进入数据集页面，通过 "Files and versions" 管理文件。

### Q: 数据集占用我的存储空间吗？
**A:** Hugging Face提供免费的数据集存储空间，公开数据集没有大小限制。

### Q: 合作者需要注册账号吗？
**A:** 如果数据集是公开的，不需要账号即可下载。如果是私有的，需要账号并获得授权。

### Q: 如何验证下载的数据完整性？
**A:** 使用以下命令：
```bash
# 检查文件夹数量
ls -d point* | wc -l  # 应该输出 10

# 检查总大小
du -sh .  # 应该约6.8GB

# 验证bag文件
rosbag info point1/pick_000.bag
```

---

## 联系方式

如有问题，请联系数据集维护者。
