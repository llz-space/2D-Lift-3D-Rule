# 2D-Lift-3D-Rule

从 **RGB-D 序列 + Camera 位姿 + 语义/实例信息** 构建场景级实例 3D 结果（支持 2D/3D 联合导出）。

## 当前能力（新）

`src/instance_bbox_pipeline.py` 现在支持两种模式：

1. **外部实例分割输入模式（推荐）**  
   - 直接接收外部 `instance_id_map + semantic_class_map`。  
   - 可设置 `use_tracking=False`，不维护轨迹，逐帧输出实例结果。  

2. **无语义输入模式（RGB-D + Camera）**  
   - 若缺失实例/语义图，使用 `DetectorSegmentor` 接口自动补全（检测 + 分割）。  
   - 你可以接入任意模型（YOLO-World/SAM2/GroundingDINO 等）实现该接口。

## 输出内容

每帧输出 `FrameOutput`，包含：
- 图片路径 `image_path`
- 深度图路径 `depth_path`
- 相机参数 `camera_intrinsics`
- 位姿 `world_T_camera`
- 实例列表 `instances`

每个实例输出 `InstanceOutput`，包含：
- `instance_id`
- `semantic_class_id`
- `bbox2d_xyxy`
- `semantic_artifact_path`（2D语义可视化相对路径）
- `obb3d`（统一世界坐标系下的 OBB，含中心、尺寸、旋转矩阵）

并可通过 `dump_outputs_json(...)` 导出 JSON（可直接作为后续视频渲染输入清单）。

## OBB 与紧邻物体问题

- 当前 3D 框已从 AABB 升级为 **PCA OBB**。  
- 轨迹模式下，关联从“仅质心距离”升级为多信号联合：  
  - 3D 中心距离  
  - 3D OBB 近似 IoU  
  - 2D bbox IoU  

这比旧逻辑更能处理“两个椅子靠得很近”的情况，但在严重遮挡/长时间交叉时仍建议升级为 Hungarian + 外观特征 ReID。

## 快速示例

```python
processor = SceneInstanceProcessor(intrinsics)
outputs = processor.process_sequence(
    frames,
    detector_segmentor=my_model,   # 仅在无语义输入时需要
    use_tracking=False,            # 外部实例输入可关闭轨迹
)
dump_outputs_json(outputs, "outputs/scene_instances.json")
```



### 关于 `DetectorSegmentor`

- `DetectorSegmentor` 是一个**模型接口协议（Protocol）**，作用是在“无语义输入”时补齐：
  - `instance_id_map`
  - `semantic_class_map`
  - 可选的 2D 语义可视化路径
- 当前仓库**没有内置** Grounded-SAM 或其他大模型权重/推理代码；你需要按工程环境自行接入。
- 我在代码里提供了 `GroundedSAMSegmentor` 占位适配器，明确了如何对接更强模型栈（Grounded-SAM / SAM2 / OWLv2+SAM）。

## 项目内置 RAG 知识库

```bash
python scripts/rag_build.py
python scripts/rag_query.py "两个 sofa 如何避免点云混合"
```

详见 `rag/README.md`。
