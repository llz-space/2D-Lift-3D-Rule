# 2D-Lift-3D-Rule

基于 `RGB-D + Camera` 的场景级 3D OBB 导出管线，支持两种输入模式：

1. 已有 `instance_id_map + semantic_class_map` 时，直接做 2D 到 3D 提升。
2. 只有 `RGB-D + Camera`、缺失语义/实例图时，使用 `Grounded-Segment-Anything` 自动补齐实例与语义，再继续 3D OBB 计算。

## 当前状态

- `SceneInstanceProcessor` 已可直接处理外部实例图或模型补全后的实例图。
- `GroundedSAMSegmentor` 不再是占位接口，现已接入：
  - `repo` 后端：优先走官方 `Grounded-Segment-Anything` 仓库里的 `GroundingDINO + SAM`
  - `transformers` 后端：在 Windows/轻量环境下回退到 Hugging Face `GroundingDINO + 官方 SAM`
  - `ram` 自动标签：可选，用于“连类别词表都没有”的场景
- 新增环境脚本、checkpoint 下载脚本、端到端 CLI、环境输出 txt 导出脚本。

## 目录

- `src/instance_bbox_pipeline.py`
  主流程、3D OBB、轨迹逻辑、JSON 导出。
- `src/grounded_sam_segmentor.py`
  Grounded-SAM 适配器。
- `scripts/run_grounded_sam_rgbd_pipeline.py`
  端到端入口：读取 RGB-D manifest，跑 Grounded-SAM，再导出 JSON。
- `scripts/setup_grounded_sam_env.ps1`
  Windows/Conda 环境搭建脚本。
- `scripts/download_grounded_sam_weights.py`
  下载 SAM 与 GroundingDINO 权重。
- `scripts/export_runtime_env.py`
  导出当前环境摘要到 `outputs/runtime_environment.txt`。
- `rag/`
  仓库内本地 RAG 知识库。

## 环境搭建

推荐用 Conda，新建独立环境，不直接复用系统 Python 3.13。

```powershell
pwsh -File scripts/setup_grounded_sam_env.ps1
```

默认行为：

- 创建 `grounded_sam_rgbd` 环境
- 安装 `PyTorch + torchvision`
- 安装 `requirements-grounded-sam.txt`
- 安装官方仓库中的 `segment_anything`
- 导出一份 `outputs/runtime_environment.txt`

如果你确认本机已具备 `Visual Studio Build Tools + CUDA`，并且希望强制启用官方 `GroundingDINO` repo 后端，再执行：

```powershell
pwsh -File scripts/setup_grounded_sam_env.ps1 -InstallRepoBackend
```

说明：

- Windows 下 `GroundingDINO` 官方 repo backend 依赖 C++/CUDA 扩展，编译门槛较高。
- 因此本项目默认建议 `backend=auto`，让它优先尝试 repo backend，失败后自动回退到 `transformers`。

## 下载权重

```powershell
conda run -n grounded_sam_rgbd python scripts/download_grounded_sam_weights.py --model-set default
```

默认会下载：

- `checkpoints/sam_vit_b_01ec64.pth`
- `checkpoints/groundingdino_swint_ogc.pth`

如需更大的 SAM：

```powershell
conda run -n grounded_sam_rgbd python scripts/download_grounded_sam_weights.py --model-set sam_h
```

## 端到端调用

先准备一个 manifest，例如：

```json
{
  "camera_intrinsics": {
    "fx": 575.0,
    "fy": 575.0,
    "cx": 319.5,
    "cy": 239.5
  },
  "frames": [
    {
      "image_path": "data/frame_000_rgb.png",
      "depth_path": "data/frame_000_depth.npy",
      "world_T_camera": [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ]
    }
  ]
}
```

再运行：

```powershell
conda run -n grounded_sam_rgbd python scripts/run_grounded_sam_rgbd_pipeline.py `
  --manifest data/scene_manifest.json `
  --class-names chair table sofa `
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth `
  --backend auto `
  --output-json outputs/scene_instances.json `
  --semantic-label-map outputs/semantic_label_map.json
```

如果你已经下载了 `groundingdino_swint_ogc.pth`，并希望优先尝试官方 repo backend：

```powershell
conda run -n grounded_sam_rgbd python scripts/run_grounded_sam_rgbd_pipeline.py `
  --manifest data/scene_manifest.json `
  --class-names chair table sofa `
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth `
  --grounded-checkpoint checkpoints/groundingdino_swint_ogc.pth `
  --backend auto
```

如果连类别词表都没有，可以切到 RAM 自动标签：

```powershell
conda run -n grounded_sam_rgbd python scripts/run_grounded_sam_rgbd_pipeline.py `
  --manifest data/scene_manifest.json `
  --backend auto `
  --auto-label-source ram `
  --ram-checkpoint checkpoints/ram_swin_large_14m.pth `
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth
```

## 输出

每帧输出 `FrameOutput`：

- `image_path`
- `depth_path`
- `camera_intrinsics`
- `world_T_camera`
- `instances`

每个实例输出 `InstanceOutput`：

- `instance_id`
- `semantic_class_id`
- `bbox2d_xyxy`
- `semantic_artifact_path`
- `obb3d`

此外建议同时保留：

- `outputs/semantic_label_map.json`
  语义 ID 到类别名映射
- `outputs/runtime_environment.txt`
  实际运行环境摘要

## 当前仍未完整实现的逻辑

本次已补齐的空接口：

- `src/instance_bbox_pipeline.py` 中原来的 `GroundedSAMSegmentor.infer()` 占位实现

仍属于“接口/增强点，但不是这次缺失实现”的部分：

- `DetectorSegmentor` 仍然保留为协议接口，这是设计选择，不是缺陷
- 轨迹关联仍未升级为 Hungarian 全局匹配
- 未加入 ReID / appearance 特征
- 未加入轨迹状态机与置信度管理

详细审计见：

- `rag/corpus/unimplemented_logic_audit.md`

## RAG

```powershell
python scripts/rag_build.py
python scripts/rag_query.py "GroundedSAM 现在支持哪些后端"
```

RAG 文档已补充本次接入说明与未实现逻辑审计。
