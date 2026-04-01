# 未实现逻辑审计（2026-04-02）

## 已补齐

- `src/instance_bbox_pipeline.py` 中的 `GroundedSAMSegmentor.infer()` 原先仅保留接口并抛出 `NotImplementedError`
- 当前已实现为真正可用的 Grounded-SAM 适配器

## 仍保留为接口/协议

- `DetectorSegmentor` 仍然是 `Protocol`
- 这是为了允许后续替换为其他检测+分割模型，不属于遗漏实现

## 仍未做的增强项

- 轨迹关联尚未升级为 Hungarian 全局匹配
- 尚未接入 ReID / appearance 特征
- 尚未引入轨迹状态机（lost / dead）和置信度管理
- RAM 自动标签是可选能力，但不属于默认环境的一部分

## 结论

当前仓库里真正“只留了接口但没实现”的主逻辑，核心就是 `GroundedSAMSegmentor`。
本次已补上。
