# Yolo Smoke Detect

吸烟识别项目仓库。Git 仓库只保存代码、配置和依赖清单；视频数据、训练数据集、模型权重、推理结果不进仓库，放在本地或服务器对应目录中。

## 目录结构

```text
Yolo-smoke-detect/
  configs/
    smoking.yaml              # 监控流程参数
    cigarette_dataset.yaml    # YOLO 数据集配置
    env.ps1                   # Windows PowerShell 环境变量
    env.cmd                   # Windows CMD 环境变量
  scripts/
    run_yolo_mark_real_samples.py
  src/
  requirements.txt            # 本地 CPU 测试环境
  requirements-server.txt     # 服务器 GPU 环境，torch 单独安装
```

本地或服务器运行时再创建这些目录：

```text
data/
  videos/                     # 原始 mp4 测试视频，不提交
  datasets/
    smoking/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
  outputs/                    # 推理输出，不提交
runs/                         # YOLO 训练结果，不提交
```

## 本地 CPU 环境

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
. .\configs\env.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 服务器 GPU 环境

服务器 A100 驱动显示 CUDA 12.2，可以优先安装 PyTorch CUDA 12.1 轮子：

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/zhanglingyimatrix-cloud/Yolo-smoke-detect.git smoking-recognition
cd smoking-recognition

conda activate my_yolo_env
python -m pip install -U pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-server.txt
```

检查 GPU 是否可用：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
PY
```

## 推理测试

先把测试视频放到：

```text
data/videos/real_samples/intel/
```

本地 CPU：

```powershell
python scripts/run_yolo_mark_real_samples.py --device cpu
```

服务器 GPU，建议先用当前空闲的 GPU 1：

```bash
python scripts/run_yolo_mark_real_samples.py --device cuda:1
```

输出会写入：

```text
data/outputs/yolo/real_samples/intel/
```

## 吸烟识别原型

本地先用姿态模型做疑似吸烟识别：

```powershell
python scripts/detect_smoking.py --source data/videos/real_samples/intel/people-detection.mp4 --device cpu
```

输出会写入：

```text
data/outputs/smoking/
```

当前没有 `data/models/cigarette.pt` 时，脚本使用 `pose_only` 模式，只判断手是否在 3 到 8 秒时间窗口内多次靠近口鼻区域，结果应理解为疑似吸烟。后续训练出香烟/打火机/烟雾小目标模型后，把权重放到：

```text
data/models/cigarette.pt
```

同一个脚本会自动切换为 `pose_object` 模式，同时要求姿态证据和小目标证据，误报会明显下降。

当前脚本还会输出 `focus_hand_to_face_frames`。这个字段表示手到脸距离足够近、值得进入细节模型复查的帧数。它使用人体尺度归一化距离，不直接用像素距离，所以能减轻近大远小问题。后续细节模型可以只在这些重点帧的人头/手部附近区域检测 `cigarette`、`smoke` 或 `lighter`。

## 深度过滤

为了减少 2D 画面中手遮挡脸但 Z 方向距离较远造成的误判，脚本支持在黄色候选区域 crop 上运行单目深度模型。当前配置使用：

```yaml
depth:
  enabled: true
  backend: transformers
  model: depth-anything/Depth-Anything-V2-Small-hf
```

安装深度依赖：

```bash
python -m pip install -r requirements-depth.txt
```

深度模块只在 `detail_trigger=true` 的候选 crop 上运行，不对全视频每一帧运行。结果会写入 JSON：

```text
depth_status              # ok / missing_dependency / model_unavailable / inference_failed
depth_consistent          # 手/估算指尖和嘴部深度是否接近
depth_delta_ratio         # 归一化深度差
depth_threshold           # 当前阈值
depth_debug_path          # 可选深度热力图
```

如果 `require_depth_consistency_for_detail=true`，并且深度模型判断为 Z 方向冲突，脚本会标记：

```text
detail_model_status: depth_conflict_skip_detail_model
```

这表示 2D 看起来靠近，但深度不一致，不作为后续强吸烟证据。

## 微调数据格式

吸烟细分类模型的数据建议采用 YOLO 检测格式，标签先从小目标开始：

```yaml
names:
  0: cigarette
  1: lighter
  2: smoke
```

后续人体姿态逻辑不建议只靠单帧分类，需要结合：

- 人体检测框
- 手、嘴、头部关键点
- 香烟/打火机/烟雾小目标检测
- 3 到 8 秒时间窗口内的连续命中

训练命令示例：

```bash
yolo detect train model=yolo11n.pt data=configs/cigarette_dataset.yaml imgsz=960 epochs=100 batch=32 device=1 project=runs/smoking name=cigarette_yolo11n
```

训练完成后重点取：

```text
runs/smoking/cigarette_yolo11n/weights/best.pt
```
