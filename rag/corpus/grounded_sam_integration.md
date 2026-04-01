# Grounded-SAM 集成备忘（2026-04-02）

## 本次接入内容

- 已将官方仓库克隆到 `third_party/Grounded-Segment-Anything`。
- `GroundedSAMSegmentor` 已从占位实现升级为真实可用适配器。
- 适配器支持三种模式：
  - `repo`：优先使用官方 GroundingDINO repo backend
  - `transformers`：回退到 Hugging Face 的 GroundingDINO 检测实现
  - `auto`：先尝试 repo，失败自动回退 transformers

## 语义缺失场景

“缺失语义信息”在当前项目里主要指：

1. 没有 `instance_id_map`
2. 没有 `semantic_class_map`

现有拉通方式：

1. 用 GroundingDINO 根据类别词表得到 2D boxes
2. 用 SAM 根据 boxes 得到 masks
3. 聚合为 `instance_id_map + semantic_class_map`
4. 再继续走原始 3D OBB 管线

## 无类别词表时

- 支持 `auto_label_source='ram'` 的可选路径
- 该路径依赖额外的 RAM 权重和 `ram` 包
- 如果没有安装 RAM，则必须显式提供 `class_names` / `text_prompt` / `classes_file`

## 环境建议

- 不要使用系统 Python 3.13
- 推荐 Conda + Python 3.10
- Windows 下 repo backend 可能因为 GroundingDINO 扩展编译失败而不可用
- 因此默认推荐运行时使用 `backend=auto`

## 输出补充

除原有 `scene_instances.json` 外，建议同时输出：

- `outputs/semantic_label_map.json`
- `outputs/runtime_environment.txt`
