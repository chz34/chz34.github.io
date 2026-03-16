---
title: MindSpore 全场景 AI 框架
date: 2024-01-03 00:00:00 +0800
categories: [工作项目, AI框架]
tags: [AI, 机器学习, 深度学习, Ascend, Python]
---

## 简介

[MindSpore](https://www.mindspore.cn) 是华为推出的全场景深度学习框架，深度适配 Ascend AI 处理器，同时支持 GPU 和 CPU，面向云、边、端全场景部署。

## 核心特点

- **自动微分**：函数式自动微分机制，支持高阶导数
- **图算融合**：将计算图中的多个算子融合为单一高效内核，减少调度开销
- **全场景部署**：统一的前端 API，面向云端训练、边缘推理、端侧部署

## 关键模块

### 自动并行

MindSpore 提供半自动和全自动并行策略，支持数据并行、算子级模型并行、流水线并行：

```python
import mindspore as ms
from mindspore import nn

# 设置并行模式
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL)
```

### 动态图 / 静态图

通过 `set_context(mode=...)` 在 PyNative（动态图）和 Graph（静态图）模式间切换，兼顾调试便捷性与执行效率。

### Ascend 适配

针对 Ascend 910/910B 等 NPU 芯片做了深度算子优化，充分发挥达芬奇架构的矩阵计算能力。

## 参与工作

主要参与方向：算子开发、图编译优化、性能调优。
