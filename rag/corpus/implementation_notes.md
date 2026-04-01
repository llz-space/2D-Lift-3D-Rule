# 实现细节备忘（2026-04）

## 关键类
- `SceneInstanceProcessor`: 主流程入口。
- `DetectorSegmentor`: 检测+分割模型接口（Protocol）。
- `FrameOutput` / `InstanceOutput`: 标准输出结构。
- `OBB`: 中心、尺寸、旋转矩阵。

## 关键能力
1. 外部实例输入可不维护轨迹（`use_tracking=False`）。
2. 缺失语义输入时可通过模型接口自动补全。
3. 输出同时保留 2D bbox / 2D语义路径 / 3D OBB / 相机与路径元信息。

## 轨迹增强点
- 相比旧版“质心距离”匹配，当前关联融合了：
  - center distance
  - approximate 3D IoU
  - 2D IoU
- 近邻同类目标（如紧挨的椅子）区分能力更好。

## 下一步建议
- 换成 Hungarian 全局匹配。
- 增加 ReID/appearance 特征。
- 引入轨迹状态机（lost/dead）和置信度管理。
