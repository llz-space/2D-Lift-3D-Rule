# Project RAG 库（本地轻量版）

这个目录用于维护项目的检索增强知识库，便于后续迭代时快速回溯设计、实现与约束。

## 目录结构

- `rag/corpus/`：知识文档（按主题维护）。
- `rag/index.json`：由脚本自动生成的倒排索引与文档元信息。
- `scripts/rag_build.py`：从 `rag/corpus/*.md` 构建索引。
- `scripts/rag_query.py`：命令行检索脚本（TF-IDF 风格打分 + 中文/英文分词）。

## 快速开始

```bash
python scripts/rag_build.py
python scripts/rag_query.py "两个 sofa 如何避免点云混合"
```

## 维护规范

1. 每次新增能力或修复问题时，补充或更新 `rag/corpus` 中对应文档。
2. 提交前执行 `python scripts/rag_build.py`，确保索引与文档同步。
3. 查询时优先命中 `rag/corpus`，减少跨文件全文搜索。

## 备注

- 这是**仓库内本地 RAG**，无外部服务依赖，适合离线/CI 环境。
- 若后续需要，可扩展到向量检索（FAISS/pgvector）并保留当前 lexical fallback。
