# 场景级实例 3D OBB 管线总览

## 输入模式
1. 外部实例分割/语义输入：`instance_id_map + semantic_class_map`。
2. 无语义输入（RGB-D + Camera）：通过 `DetectorSegmentor` 自动生成实例/语义。

## 核心流程
1. 逐实例 mask 反投深度得到相机系点云。
2. 根据位姿变换到统一世界坐标系。
3. 计算每个实例的 OBB（PCA）。
4. 记录每帧元数据：图片路径、深度路径、相机参数、位姿。
5. 导出每个实例的 2D bbox、2D语义可视化路径、3D OBB。

## 轨迹逻辑
- 可选开启 `use_tracking=True`。
- 若关闭轨迹，直接按输入实例逐帧输出（适合外部实例分割已稳定场景）。
- 若开启轨迹，使用多信号关联：3D中心距离 + 3D OBB IoU + 2D bbox IoU。

## 输出建议
- 使用 `dump_outputs_json` 统一导出后，可直接驱动视频渲染或数据集打包。
