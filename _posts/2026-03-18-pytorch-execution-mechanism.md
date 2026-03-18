---
title: PyTorch 执行机制全景：从单算子调用到分布式编译
date: 2026-03-18 00:00:00 +0800
categories: [AI框架, PyTorch]
tags: [PyTorch, 深度学习, torch.compile, CUDA, 分布式, FSDP]
---

> 本文系统梳理 PyTorch 的完整执行机制，覆盖计算对象的抽象层级、不同调用方式下的执行路径、以及各类捕获与下发机制的工作原理及相互关系。

---

## 目录

1. [计算对象的抽象层级](#1-计算对象的抽象层级)
2. [三种基本调用方式](#2-三种基本调用方式)
3. [C++ Dispatcher：所有路径的汇聚点](#3-c-dispatcher所有路径的汇聚点)
4. [torch.jit：静态编译（已废弃）](#4-torchjit静态编译已废弃)
5. [torch.fx.Tracer：图变换的基础设施](#5-torchfxtracer图变换的基础设施)
6. [torch.compile：现代运行时 JIT](#6-torchcompile现代运行时-jit)
7. [FSDP：基于 Module Hook 的分布式并行](#7-fsdp基于-module-hook-的分布式并行)
8. [CUDA Graph：Driver 层的 Kernel 录制与回放](#8-cuda-graphdriver-层的-kernel-录制与回放)
9. [各机制之间的关系与组合](#9-各机制之间的关系与组合)
10. [完整执行路径总览](#10-完整执行路径总览)

---

## 1. 计算对象的抽象层级

PyTorch 中的计算对象构成一个从高到低的抽象栈，每一层都有明确的职责边界。

```
┌─────────────────────────────────────────────┐
│  nn.Module                                  │  Python 层：有状态的计算单元
│    └─ parameters (nn.Parameter)             │
│    └─ forward() / hooks                     │
├─────────────────────────────────────────────┤
│  Python Function / Callable                 │  Python 层：无状态计算逻辑
├─────────────────────────────────────────────┤
│  fx.Graph / fx.GraphModule                  │  Python IR 层：可检查/修改的图表示
├─────────────────────────────────────────────┤
│  torch.ops.aten.xxx.default (OpOverload)    │  Python-C++ 边界：ATen 算子
│    └─ OpOverloadPacket (torch.ops.aten.mm)  │
├─────────────────────────────────────────────┤
│  C++ Dispatcher                             │  C++ 层：按 DispatchKey 路由
│    └─ DispatchKeySet 优先级队列             │
├─────────────────────────────────────────────┤
│  Kernel                                     │  C++/CUDA 层：实际计算实现
│    └─ CPU kernel (ATen/native)              │
│    └─ CUDA kernel (.cu / Triton .py)        │
└─────────────────────────────────────────────┘
```

### 1.1 ops 与 aten ops

**`torch.ops`** 是 PyTorch 中所有算子的命名空间入口。调用 `torch.ops.aten.mm.default` 等价于在 Python 侧触发一个命名为 `aten::mm` 的 C++ Dispatcher 调用。

每个算子有两种对象形态：

- **`OpOverloadPacket`**（`torch.ops.aten.mm`）：算子族，可根据 `.default` / `.out` 等重载名进一步索引
- **`OpOverload`**（`torch.ops.aten.mm.default`）：具体的一个重载，是真正参与 Dispatcher 路由的对象

```python
# 两者等价：
torch.mm(a, b)                          # 高层 Python API
torch.ops.aten.mm.default(a, b)        # 直接通过 OpOverload 调用
```

`torch.mm` 内部最终也会走到 `aten::mm` 的 Dispatcher 调用，区别只是入口的 Python 包装层数不同。

### 1.2 kernel

Kernel 是 Dispatcher 路由链的终点，是实际执行数值计算的 C++/CUDA 代码。每个算子按 `DispatchKey` 注册多个 kernel：

```cpp
// 在 C++ 中注册 kernel（以 ATen native 为例）
TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("mm", &at::native::mm_cpu);
}
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
    m.impl("mm", &at::native::mm_cuda);
}
```

一个 ATen 算子可以有多个 kernel 注册：
- `CPU` kernel：在 `aten/src/ATen/native/` 下的 C++ 实现
- `CUDA` kernel：在 `aten/src/ATen/native/cuda/` 下的 `.cu` 文件
- `Meta` kernel：只推断形状，不执行计算（用于 FakeTensor）
- `CompositeImplicitAutograd` kernel：用纯 ATen 算子组合实现，自动获得所有设备支持
- `Autograd` kernel：自动微分的正反向定义（在 `tools/autograd/derivatives.yaml`）

### 1.3 nn.Module

`nn.Module` 是 PyTorch 中有**状态**的计算单元。它的核心能力：

- 持有 `nn.Parameter`（可训练参数，本质是带 `requires_grad=True` 的 Tensor）
- 通过 `forward()` 定义计算逻辑
- 提供 `register_forward_pre_hook` / `register_forward_hook` / `register_backward_hook` 等 hook 接口
- 支持 `state_dict()` / `load_state_dict()` 序列化
- 支持 `named_parameters()` / `named_modules()` 递归遍历

Module 本质上是**函数 + 状态的封装**，`module(x)` 等价于 `module.forward(x)`（通过 `__call__` 分发，并在前后执行所有注册的 hook）。

### 1.4 Python Function

Python 函数是无状态的计算逻辑，可以调用任意 `torch.ops` 或 Module。从 PyTorch 执行机制的角度，一个 Python 函数和一个 Module 的 `forward` 方法没有本质区别——都是 Python 代码，最终通过相同的 Dispatcher 路径落到 kernel。

更重要的是，**Python 函数（Callable）是 PyTorch 所有编译和捕获 API 的统一接口**。`torch.compile`、`torch.fx.symbolic_trace`、`make_fx`、`torch.jit.trace/script` 均以 callable 为输入，编译产物同样是 callable。整个编译流水线是 callable → callable 的变换链（详见 2.2 节）。`nn.Module` 通过 `__call__` 实现 callable 协议，因此对 Dynamo 等机制而言，Module 和普通函数是同一类对象。

### 1.5 fx.Graph 与 fx.GraphModule

`fx.Graph` 是 PyTorch 中 Python 层的图 IR（中间表示）。它由 `Node` 构成，每个 `Node` 记录一次操作：

| Node 类型 | 含义 | 示例 |
|---|---|---|
| `placeholder` | 函数输入 | `x: Tensor` |
| `call_function` | 调用 torch 函数 | `aten.mm.default(x, w)` |
| `call_method` | 调用 Tensor 方法 | `x.view(4, -1)` |
| `call_module` | 调用 nn.Module | `self.linear(x)` |
| `get_attr` | 访问属性 | `self.weight` |
| `output` | 返回值 | `return y` |

`fx.GraphModule` 是 `nn.Module` 的子类，持有一张 `fx.Graph`，并从这张图**自动生成** Python `forward()` 方法代码。这意味着 `fx.GraphModule` 同时是一个可运行的 Module 和一个可检查/修改的图表示。

```python
gm = torch.fx.symbolic_trace(my_fn)
print(gm.graph)     # 查看图结构
gm.graph.print_tabular()  # 打印表格
# 修改后必须重新编译：
gm.recompile()      # 重新生成 forward() 方法
```

---

## 2. 三种基本调用方式

### 2.1 单算子直接调用

```python
result = torch.mm(a, b)
# 等价于：
result = torch.ops.aten.mm.default(a, b)
```

执行路径：

```
torch.mm(a, b)
    │
    ▼ Python binding (torch/_C/_VariableFunctions.pyi)
    │
    ▼ C++ Dispatcher
    │   ├─ 收集 DispatchKeySet（来自 a, b 的设备类型 + 当前 TLS 中的 mode 栈）
    │   ├─ 按优先级查找第一个有 kernel 的 key
    │   └─ 调用对应 kernel
    │
    ▼ aten::mm 的 CUDA/CPU kernel
```

这是最直接的路径，没有任何 Python 捕获层。Dispatcher 在 C++ 侧完成所有路由。

### 2.2 函数中调用

```python
def my_fn(x, w):
    return torch.nn.functional.relu(x @ w)

result = my_fn(x, w)
```

从执行机制角度，这和单算子调用完全相同——Python 解释器顺序执行每行代码，每个 `torch` 算子调用都独立走一次 Dispatcher 路由。Python 函数本身不提供任何额外的捕获或优化。

**关键特性**：
- 每个算子调用都有独立的 CPU launch overhead（启动 kernel 的 CPU 开销）
- Python 控制流（if/for/while）完全在 CPU 侧执行，不进入任何图中
- 无法跨算子进行内核融合优化

**Python 函数是后续所有捕获机制的基础对象。** 上面描述的是函数的"朴素执行"语义，但函数（Callable）同时也是 PyTorch 所有编译和图捕获 API 的统一输入接口：

```python
# 以下 API 都以 callable 为第一参数：
torch.compile(my_fn)                  # Dynamo 字节码拦截
torch.fx.symbolic_trace(my_fn)        # fx.Tracer 符号执行
torch.jit.trace(my_fn, example_inputs)# JIT trace 录制
torch.jit.script(my_fn)               # JIT 静态编译
make_fx(my_fn)(x, w)                  # make_fx ATen 级追踪
```

这种统一性来自 Python 的 callable 协议：`nn.Module` 实例通过 `__call__` 也是 callable；`torch.compile` 的返回值同样是 callable；AOTAutograd 中 `fw_compiler` 的参数是 `fx.GraphModule`（nn.Module 子类，因此也是 callable），其返回值也必须是 callable。整个编译流水线可以理解为**callable → callable 的变换链**：

```
用户函数（callable）
    │
    ▼ torch.compile → 返回包装后的 callable（内部含 Dynamo 拦截逻辑）
    │
    ▼ Dynamo 首次调用时提取子图 → fx.GraphModule（callable）
    │
    ▼ AOTAutograd(fw_compiler=...) → 调用 fw_compiler(gm, inputs) → callable
    │
    ▼ Inductor 编译 → kernel_launcher（callable，最终调用 cubin）
```

**`nn.Module` 与函数的关系**：`module.forward` 本质上就是一个 bound method（函数），`module(x)` 经过 `__call__` 最终调用的也是这个函数。`torch.compile(model)` 在内部实际拦截的是 `model.__call__` 的字节码。因此 Module 调用和函数调用在 Dynamo 看来是同一种对象。

**`torch.autograd.Function`**：自定义反向传播时，用户通过继承 `torch.autograd.Function` 并实现 `forward`/`backward` 静态方法来定义自定义算子的前向和反向逻辑。这里的 `forward`/`backward` 同样是函数，在 AOTAutograd 的联合图追踪中会被识别为自定义 autograd node 并保留在反向图中：

```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * (x > 0).float()
```

### 2.3 Module 中调用

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, x):
        return torch.relu(self.linear(x))

model = MyModel()
result = model(x)
```

`model(x)` 实际调用的是 `nn.Module.__call__`，它在 `forward()` 前后执行所有注册的 hook：

```
model(x)
    │
    ▼ nn.Module.__call__
    │   ├─ _call_impl() → 执行所有 forward_pre_hooks
    │   ├─ self.forward(x)   ← 用户定义的计算
    │   └─ 执行所有 forward_hooks / backward_hooks
    │
    ▼ forward() 内部的每个算子调用 → Dispatcher
```

Module 调用和函数调用在 Dispatcher 层面没有区别，区别在于 hook 机制提供了"在 forward 前后注入逻辑"的能力——FSDP 正是利用这一点实现参数的 all-gather 和 reduce-scatter。

---

## 3. C++ Dispatcher：所有路径的汇聚点

Dispatcher 是 PyTorch 执行机制的核心枢纽。无论上层是函数调用、Module 调用、traced 图执行还是 Inductor 生成的 kernel launcher，最终都会经过 Dispatcher。

### 3.1 DispatchKey 体系

每个 Tensor 携带一个 `DispatchKeySet`，记录该 Tensor 相关的所有 key（如 `CUDA`、`Autograd`、`Python`）。调用一个算子时，Dispatcher 合并所有输入 Tensor 的 `DispatchKeySet` 以及当前线程局部状态（TLS）中的 key（如当前活跃的 `TorchDispatchMode`），按优先级找到第一个有注册 kernel 的 key：

```
DispatchKey 优先级（从高到低，部分关键 key）：

PreDispatch               ← torch.export 的 pre-dispatch 追踪
Python                    ← TorchDispatchMode（__torch_dispatch__ hook）
FuncTorchDynamicLayerFront ← functorch 变换前端
Autograd                  ← 自动微分（记录操作到 autograd graph）
AutocastCPU / AutocastCUDA ← 自动混合精度
BackendSelect             ← 按输入选择后端设备
CUDA                      ← CUDA kernel（物理 kernel 注册点）
CPU                       ← CPU kernel
Meta                      ← 形状推断 kernel（FakeTensor 用）
CompositeImplicitAutograd ← 算子分解实现（所有设备通用）
```

### 3.2 典型路由路径

**普通 CUDA Tensor 前向计算（无 autograd）：**

```
torch.mm(a, b)  [a, b 在 CUDA，requires_grad=False]
    │
    DispatchKeySet = {CUDA, AutogradCUDA}
    最高优先有 kernel 的 key = CUDA
    │
    ▼ CUDA kernel: at::native::mm_cuda(a, b)
```

**有 autograd 的路径：**

```
torch.mm(a, b)  [requires_grad=True]
    │
    最高 key = Autograd
    │
    ▼ Autograd kernel:
    │   记录操作到 autograd graph（Node: MmBackward）
    │   调用 redispatch(CUDA, mm, a, b)
    │
    ▼ CUDA kernel
```

**TorchDispatchMode 激活时（如 FakeTensorMode）：**

```
torch.mm(a, b)  [FakeTensorMode 激活]
    │
    TLS 中有 Python key
    │
    ▼ Python dispatch → TorchDispatchMode.__torch_dispatch__(mm, ...)
    │   FakeTensorMode：用 Meta kernel 推断形状，返回 FakeTensor
    │   ProxyTorchDispatchMode：记录到 fx.Graph，返回 Proxy
```

### 3.3 `__torch_function__` 与 `__torch_dispatch__` 的区别

这两个协议都允许 Python 代码拦截算子调用，但拦截层次不同：

| 协议 | 拦截层次 | 触发时机 | 主要用途 |
|---|---|---|---|
| `__torch_function__` | Python API 层（torch.mm 等） | 在进入 Dispatcher 之前 | fx.Tracer 的 Proxy、自定义 Tensor 子类 |
| `__torch_dispatch__` | Dispatcher C++ 层（ATen op） | 在 Dispatcher 路由后、kernel 执行前 | FakeTensor、ProxyTorchDispatchMode、TorchDispatchMode |

`__torch_function__` 捕获的是 Python API 调用（`torch.matmul`、`F.relu`），**不会捕获 ATen op 内部的其他 ATen 调用**。`__torch_dispatch__` 捕获的是 ATen 级别的每一个 op，粒度更细，信息更完整。

---

## 4. torch.jit：静态编译（已废弃）

> **状态：PyTorch 2.5+ 已标注 deprecated，不建议在新代码中使用。**

`torch.jit` 提供两种捕获方式，产出 TorchScript IR（C++ bytecode），**不透明，不可在 Python 侧修改**。

### 4.1 torch.jit.trace

运行时录制：用真实 Tensor 执行一遍函数，录制所有操作。

```python
traced = torch.jit.trace(fn, example_inputs=(x, w))
```

**机制**：用 `TraceTensor`（Tensor 子类）替换输入，`__torch_function__` 拦截所有算子调用，记录成 TorchScript 图。

**根本缺陷**：
- 只录制一次执行路径，数据相关控制流（`if tensor > 0`）被"固化"
- 不支持动态形状（录制时的 shape 被固定）
- 产出 C++ bytecode，无法在 Python 侧做图变换

### 4.2 torch.jit.script

静态解析：在装饰时解析 Python AST，编译为 TorchScript。

```python
@torch.jit.script
def fn(x: torch.Tensor, cond: bool) -> torch.Tensor:
    if cond:
        return torch.relu(x)
    return x
```

**机制**：解析 Python 源码 AST，要求严格类型注解，编译为 TorchScript IR。

**缺陷**：对 Python 语法有严格限制（不能用 `*args`、动态属性等），不支持任意 Python 代码。

---

## 5. torch.fx.Tracer：图变换的基础设施

`torch.fx.Tracer` 是 PyTorch 图变换生态的基础。它不是一个独立的执行路径，而是一套**把 Python 代码变成可操作 Graph 的工具**，被 Dynamo、AOTAutograd、量化、剪枝等所有需要图操作的工具共同使用。

### 5.1 符号执行机制：`__torch_function__` + Proxy

`fx.Tracer.trace()` 的工作原理：用 `Proxy` 对象替换真实 Tensor，然后**真正执行一遍**被追踪的函数。当 Proxy 参与任何 torch 算子调用时，`Proxy.__torch_function__` 被触发，操作被记录为 Graph 中的一个 Node，并返回新的 Proxy 对象继续流转。

```python
# fx/proxy.py:739
class Proxy:
    @classmethod
    def __torch_function__(cls, orig_method, types, args, kwargs):
        # orig_method = torch.matmul, F.relu, operator.matmul 等 Python API 层对象
        tracer = ...  # 从 args 中找到 tracer
        return tracer.create_proxy("call_function", orig_method, args, kwargs)
        # 返回新 Proxy，代表这次计算的"结果"
```

**拦截点是 Python API 层**，所以 `symbolic_trace` 产出的节点是 `torch.matmul`、`F.relu`，而非 `aten.mm.default`。

### 5.2 is_leaf_module：控制追踪深度

`Tracer.call_module()` 决定一个 `nn.Module` 是否继续展开追踪：

```python
# fx/_symbolic_trace.py:471
def is_leaf_module(self, m, module_qualified_name):
    # 默认：torch.nn 下的模块（如 nn.Linear）是 leaf，不展开
    return m.__module__.startswith("torch.nn") and not isinstance(m, nn.Sequential)
```

- **leaf module** → 记录为 `call_module` 节点，不追踪内部
- **非 leaf** → 递归执行 `forward()`，展开所有内部算子

### 5.3 fx.Tracer 的根本局限

由于机制是"执行一遍 Python 代码"，遇到数据相关控制流会直接报错：

```python
def bad_fn(x):
    if x.sum() > 0:    # ← Proxy 的布尔求值 → 报错
        return x.relu()
    return x

torch.fx.symbolic_trace(bad_fn)  # 抛出 TraceError
```

这是 `fx.Tracer` 的本质局限，也是 Dynamo 出现的动机。

### 5.4 make_fx / ProxyTorchDispatchMode：`__torch_dispatch__` 层的 Tracer

`make_fx`（`torch/fx/experimental/proxy_tensor.py`）是 fx.Tracer 的增强版本，拦截点下沉到 **Dispatcher 的 `__torch_dispatch__` 层**：

```python
# proxy_tensor.py:1772
class ProxyTorchDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func: OpOverload, types, args, kwargs):
        # func 已经是 aten.mm.default，而非 torch.matmul
        return self.tracer.create_proxy("call_function", func, args, kwargs)
```

`make_fx` 的优势：
- 产出节点是 ATen 级别（`aten.mm.default`，而非 `torch.matmul`）
- 可配合 `FakeTensor` 做形状推断（`tracing_mode="fake"`）
- 可配合 `SymInt` 做符号形状推断（`tracing_mode="symbolic"`）
- 支持 decomposition table（把高级算子分解为基础 ATen 算子）

**AOTAutograd 内部就是用 `make_fx` 追踪联合前向+反向图。**

---

## 6. torch.compile：现代运行时 JIT

`torch.compile` 是 PyTorch 2.0 引入的现代编译路径，由三个子系统串联：

```
torch.compile(model)
    │
    ▼ Dynamo（字节码拦截，图捕获）
    │
    ▼ AOTAutograd（自动微分图拆分）
    │
    ▼ Inductor（kernel 生成 + 优化）
    │
    ▼ Triton / C++ kernel
```

### 6.1 Dynamo：PEP 523 字节码拦截

Dynamo 通过 **PEP 523**（`_PyEval_SetEvalFrameFunc`）在 CPython 帧求值前插入 hook，分析 Python 字节码，将可编译的连续片段提取为 `fx.GraphModule`。

**与 `fx.Tracer` 的关键区别**：

| | fx.Tracer | Dynamo（SubgraphTracer） |
|---|---|---|
| 拦截机制 | `__torch_function__`（Python API 层） | PEP 523 字节码分析 |
| 产出节点级别 | `torch.matmul`, `F.relu`（Python API） | `aten.mm.default`（ATen） |
| 数据相关控制流 | ❌ 报错 | ✅ 图断裂（graph break）处理 |
| 动态形状 | ❌ | ✅（ShapeEnv + SymInt） |

Dynamo 内部的 `SubgraphTracer` 继承自 `fx.Tracer`，复用其建图基础设施，但由字节码分析驱动（而非直接执行 Python 函数）：

```python
# torch/_dynamo/output_graph.py:3187
class SubgraphTracer(fx.Tracer):
    def __init__(self, output_graph, ...):
        super().__init__()
        self.graph = torch.fx.Graph()
        self.input_name_to_proxy: dict[str, fx.Proxy] = {}
        self.lifted_freevars: dict[fx.Proxy, fx.Proxy] = {}
        # 支持嵌套 HigherOrderOperator 追踪
```

**图断裂（Graph Break）**：遇到 Dynamo 无法追踪的 Python 代码（数据相关控制流、不支持的 Python 操作等），Dynamo 将当前已捕获的子图编译，返回 Python 解释器执行无法追踪的部分，再继续捕获后续子图。

```
model(x):
    y = relu(x @ w)     ← 捕获为子图 1
    if y.sum() > 0:     ← graph break：退回 Python 执行
        z = y + 1       ← 捕获为子图 2
    return z
```

### 6.2 AOTAutograd：前向+反向图联合追踪

AOTAutograd（`torch/_functorch/_aot_autograd/`）接受 Dynamo 产出的 `fx.GraphModule`，通过 `make_fx` 追踪联合前向+反向计算图：

```
前向图 (gm)
    │
    ▼ make_fx 追踪联合图 (joint graph)
    │   includes: forward + backward computation
    │
    ▼ partition_fn (default_partition)
    │   拆分为：forward graph + backward graph
    │
    ▼ fw_compiler(forward_gm)   → 前向编译产物
    ▼ bw_compiler(backward_gm) → 反向编译产物
```

AOTAutograd 的两个关键贡献：
1. **把反向传播也变成图**：通过追踪 `loss.backward()` 的执行来建立反向图
2. **图级别的前向-反向划分**：`partition_fn` 决定哪些中间结果需要 save_for_backward，实现内存高效的 checkpointing

### 6.3 Inductor：图到 kernel 的编译

Inductor（`torch/_inductor/`）接受 AOTAutograd 输出的 `fx.GraphModule`，执行：

1. **图级别优化（FX passes）**：算子融合、常量折叠、死代码消除
2. **Lowering**：将 ATen op 降低为 Inductor IR
3. **Codegen**：生成 Triton（GPU）或 C++（CPU）kernel 代码
4. **编译**：调用 Triton compiler 或 gcc/clang 编译

```python
# torch/_inductor/__init__.py
def compile(gm: torch.fx.GraphModule, example_inputs, options=None):
    from .compile_fx import compile_fx
    return compile_fx(gm, example_inputs, config_patches=options)
```

Inductor 的核心优化能力：
- **水平融合**：把多个 pointwise 算子合并为一个 kernel
- **垂直融合（epilogue fusion）**：把 matmul 后的 activation 融入 matmul kernel
- **内存布局优化**：选择最优的 tensor 内存排布以减少内存带宽压力

### 6.4 ShapeEnv 与动态形状

为了让同一个编译结果服务于多种输入形状，Dynamo 使用 `ShapeEnv`（`torch/fx/experimental/symbolic_shapes.py`）将具体 shape 替换为 `SymInt` 符号变量，并在 guard 中记录形状约束：

```python
# 编译时：x.shape = [SymInt("s0"), 8]
# 生成 guard：lambda s0: s0 > 0 and s0 < 1024
# 运行时：检查 guard，若满足则复用已编译 kernel
```

---

## 7. FSDP：基于 Module Hook 的分布式并行

FSDP2（`torch/distributed/fsdp/_fully_shard/`）是 PyTorch 的全分片数据并行实现，其并行逻辑**完全基于 `nn.Module` 的 hook 机制**，与 Dispatcher、fx.Tracer、Dynamo 均无直接关系。

### 7.1 为什么必须是 nn.Module

`fully_shard` 的 API 只接受 `nn.Module`（或其 List）：

```python
# _fully_shard.py:58
def fully_shard(module: nn.Module, ...) -> FSDPModule: ...
```

FSDP 依赖 `nn.Module` 的三项基础能力：
1. `register_forward_pre_hook` / `register_forward_hook`：在 forward 前后注入 all-gather / reshard
2. `named_parameters()`：遍历需要分片的参数
3. `module.__class__ = new_cls`：动态注入 `FSDPModule` mixin（改写 MRO）

普通函数或 `torch.compile` 产物均不具备这些接口。

### 7.2 执行机制：前向 all-gather，反向 reduce-scatter

**初始化**（`fully_shard(model)` 时）：

```
1. 把 model.parameters() 转为 DTensor（Shard(0) 放置）——参数被分片
2. 动态改写 model.__class__ = FSDPLinear(FSDPModule, Linear)
3. 注册 forward pre-hook: _pre_forward → all-gather
4. 注册 forward hook:  _post_forward → reshard（可选）
5. 在输出 tensor 上注册 autograd hook → 触发 reduce-scatter
```

**前向执行流（_fsdp_state.py）**：

```
model(x)
    │
    ▼ nn.Module.__call__ → forward_pre_hooks
    │   _pre_forward():
    │     FSDPParamGroup.pre_forward()
    │       unshard() → foreach_all_gather()   ← 从各 worker 收集分片参数
    │       wait_for_unshard()                  ← 等待通信完成
    │
    ▼ model.forward(x)  [使用完整参数执行计算]
    │
    ▼ forward_hooks
    │   _post_forward():
    │     FSDPParamGroup.post_forward()
    │       reshard() → _to_sharded()           ← 释放完整参数，恢复分片状态
```

**反向执行流**：

```
loss.backward()
    │
    ▼ autograd graph 中的 RegisterPostBackwardFunction
    │   pre_backward():
    │     unshard() → all-gather 参数（再次收集用于梯度计算）
    │
    ▼ 正常反向传播（在完整参数上计算梯度）
    │
    ▼ root post-backward callback
    │   post_backward():
    │     foreach_reduce() → reduce-scatter 梯度  ← 梯度分片并聚合
    │     reshard()                               ← 释放完整参数
```

### 7.3 与 torch.compile 的组合

FSDP 和 `torch.compile` 的正确组合顺序是：**先 FSDP，再 compile**。

```python
fully_shard(model)      # 先分片参数，注入 FSDPModule hook
torch.compile(model)    # 再编译：model 仍是 nn.Module，可正常编译
```

Dynamo 对 FSDP hook 的处理由 `skip_fsdp_hooks` 控制：

```python
# torch/_dynamo/config.py:413
skip_fsdp_hooks = True   # 默认：把 FSDP hook（all-gather/reduce-scatter）当黑盒跳过
                         # 设为 False（Traceable FSDP2）：把通信操作也编译进图中
```

### 7.4 通信流重叠：多个 CUDA Stream

FSDP 通过多个专用 CUDA Stream 实现计算-通信重叠：

```python
# FSDPCommContext 中：
all_gather_copy_in_stream   # 异步拷贝输入到通信 buffer
all_gather_stream           # 执行 all-gather（与下一层前向计算重叠）
reduce_scatter_stream       # 执行梯度 reduce-scatter
all_reduce_stream           # HSDP 场景下的 all-reduce
```

---

## 8. CUDA Graph：Driver 层的 Kernel 录制与回放

CUDA Graph 的工作层次是所有 PyTorch 机制中最低的——**CUDA Driver API 层**，完全绕过 Python 解释器、Dispatcher 和任何 Python 图表示。

### 8.1 本质：录制 GPU 指令序列

```python
# cudagraphs.py:283（简化版，真实代码见 cudagraph_trees.py）
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    static_outputs = model(static_inputs)  # model = 已编译的 kernel launcher
# 此后：
graph.replay()  # CPU 发一条命令，GPU 重放全部 kernel
```

在 `torch.cuda.graph()` 上下文内，NVIDIA CUDA Driver 调用 `cudaStreamBeginCapture`，将所有 `cudaLaunchKernel`、`cudaMemcpyAsync` 等 CUDA API 调用录制为一个 DAG（有向无环图）。录制结束后，`graph.replay()`（即 `cudaGraphLaunch`）以**零 CPU 开销**重放全部 GPU 指令。

**capture 的对象不是 Python 函数或 fx.Graph，而是已编译 kernel 的启动序列。**

### 8.2 在 torch.compile 流水线中的位置

```
Dynamo → AOTAutograd → Inductor → 已编译的 Triton kernel launcher
                                                    │
                                    cudagraphify_impl() 包装
                                                    │
                                    第一次调用时：
                                      torch.cuda.graph(...):
                                          kernel_launcher(static_inputs)
                                      ← 录制为 CUDAGraph
                                    后续调用：
                                      copy inputs → graph.replay()
```

Inductor 中通过 `config.triton.cudagraphs = True` 启用（默认需手动开启或通过 `mode="reduce-overhead"` 自动启用）。

### 8.3 CUDAGraph Trees：支持 Dynamo 图断裂

标准 CUDA Graph 只支持线性执行（A → B → C 必须按顺序）。Dynamo 的图断裂会产生多个子图，需要树形结构支持。

`CUDAGraphTreeManager`（`torch/_inductor/cudagraph_trees.py`）的关键能力：

1. **共享内存池**：多个 graph 共享同一 CUDA memory pool，避免跨图 copy 中间结果
2. **树形拓扑**：允许在执行完 graph A 后，根据运行时条件执行 graph B 或 graph B'
3. **AllocatorState checkpoint**：录制新子图前，恢复 caching allocator 的内存账本状态

```
CUDAGraphTreeManager:
  root
  ├─ CUDAGraphNode(forward_graph_1)
  │     ├─ CUDAGraphNode(backward_graph_1a)  ← 某种输入形态
  │     └─ CUDAGraphNode(backward_graph_1b)  ← 另一种输入形态（动态形状）
  └─ CUDAGraphNode(forward_graph_2)
```

### 8.4 CUDA Graph 的约束

以下情况会自动 fallback 到非 graph 执行：

- 图内有 CPU-GPU 同步（如 `.item()`、`.cpu()`）
- 图内有数据相关的内存分配（不能用固定 buffer）
- 有不兼容的算子（某些 NCCL 通信算子、cuBLAS workspace 不固定的算子）
- 输入有 mutation（in-place 操作改变输入 storage 地址）
- 多设备操作

---

## 9. 各机制之间的关系与组合

### 9.1 拦截层次对比

所有机制的本质区别在于**在哪一层拦截计算**：

```
Python 代码（用户写的 model(x)）
    │
    │  ← [Dynamo] PEP 523 字节码层拦截
    │       分析字节码，提取 fx.GraphModule
    │
    ▼
Python API 调用（torch.mm, F.relu, operator.matmul）
    │
    │  ← [fx.Tracer + __torch_function__] Python API 层拦截
    │       Proxy 替换 Tensor，记录操作为 fx.Node
    │
    ▼
C++ Dispatcher（DispatchKey 路由）
    │
    │  ← [__torch_dispatch__ / TorchDispatchMode] Dispatcher 层拦截
    │       FakeTensorMode, ProxyTorchDispatchMode (make_fx)
    │       TorchDispatchMode（HWDispatchMode 等自定义 mode）
    │
    │  ← [FSDP hook] Module 层拦截（不在算子路径上，在 forward 调用前后）
    │
    ▼
ATen kernel（CPU/CUDA）
    │
    ▼
CUDA Driver API
    │
    │  ← [CUDA Graph] Driver 层拦截
    │       录制 cudaLaunchKernel 序列
    │
    ▼
GPU 执行
```

### 9.2 各机制产出物与消费者

```
调用方式         产出                消费者
─────────────────────────────────────────────────
fx.Tracer   →   fx.GraphModule   →  用户自定义 pass, Inductor
make_fx     →   fx.GraphModule   →  AOTAutograd, torch.export
Dynamo      →   fx.GraphModule   →  AOTAutograd → Inductor
AOTAutograd →   fw+bw Graph      →  Inductor (fw_compiler/bw_compiler)
Inductor    →   kernel launcher  →  直接执行 / CUDA Graph 封装
CUDA Graph  →   CUDAGraph 对象   →  graph.replay() 直接执行
FSDP        →   FSDPModule       →  torch.compile 可进一步编译
torch.jit   →   ScriptModule     →  C++ 部署（Legacy）
```

### 9.3 典型组合：torch.compile + FSDP + CUDA Graph

```python
model = MyTransformer()

# 步骤 1：FSDP 分片（Module hook 层）
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)

# 步骤 2：torch.compile（Dynamo + AOTAutograd + Inductor）
# Inductor 内部自动启用 CUDA Graph（mode="reduce-overhead"）
compiled_model = torch.compile(model, mode="reduce-overhead")

# 运行时路径：
# model(x)
#   → FSDP pre-hook: all-gather 参数（CUDA Stream）
#   → Dynamo 已编译子图（Triton kernel launcher）
#     → CUDA Graph replay（零 CPU overhead）
#   → FSDP post-hook: reshard 参数
```

### 9.4 fx.Graph 是贯通 compile 流水线的统一 IR

```
用户代码
    │
    ▼ Dynamo          → fx.GraphModule  (ATen 节点，带 meta/guards)
    │
    ▼ AOTAutograd     → fw fx.Graph + bw fx.Graph  (ATen 节点，形状已推断)
    │
    ▼ Inductor FX pass → 优化后的 fx.GraphModule
    │
    ▼ Inductor Lowering → Inductor IR（不再是 fx.Graph）
    │
    ▼ Triton codegen   → .py kernel 文件
    │
    ▼ Triton 编译      → cubin（GPU 二进制）
```

`fx.GraphModule` 是整个编译流水线中 Python 可见、可操作的 IR 形态。进入 Inductor lowering 后就变成 Inductor 的内部 IR，不再对外暴露为 `fx.Graph`。

---

## 10. 完整执行路径总览

### 10.1 Eager 执行（无 compile，无 FSDP）

```
model(x)
    │ nn.Module.__call__
    ├─ forward_pre_hooks（若有）
    ├─ model.forward(x)
    │       ├─ self.linear(x)
    │       │       ├─ nn.Module.__call__ → Linear.forward
    │       │       │   F.linear(x, self.weight, self.bias)
    │       │       │   → torch.ops.aten.addmm.default(bias, x, weight.T)
    │       │       │   → C++ Dispatcher → CUDA kernel
    │       │       └─ ←
    │       ├─ torch.relu(y)
    │       │   → torch.ops.aten.relu.default(y)
    │       │   → C++ Dispatcher → CUDA kernel
    │       └─ return z
    └─ forward_hooks（若有）
返回 z
```

每个算子独立走一次 CPU launch，无任何跨算子优化。

### 10.2 torch.compile 执行（首次调用，触发编译）

```
compiled_model(x)
    │ Dynamo 截获字节码
    │
    ├─ 追踪执行 → SubgraphTracer 建图
    │     记录所有 aten op 为 fx.Node
    │
    ├─ 调用 backend(gm, example_inputs)
    │     └─ AOTAutograd:
    │           make_fx 追踪联合图
    │           partition_fn 拆分 fw/bw
    │           fw_compiler(fw_gm) → Inductor:
    │               FX passes（融合、优化）
    │               Lowering → Inductor IR
    │               Triton codegen → kernel.py
    │               编译 → cubin
    │               返回 kernel_launcher
    │
    └─ 返回 compiled_fn（包装了 kernel_launcher）

compiled_fn(x)  [后续调用，直接执行]
    │ 检查 guards（shape、dtype、值范围等）
    ├─ 若 guards 通过 → 直接调用 kernel_launcher(x)
    └─ 若 guards 失败 → 重新编译（recompile）
```

### 10.3 torch.compile + CUDA Graph（mode="reduce-overhead"）

```
compiled_model(x)  [首次 graph 录制]
    │
    ├─ Inductor 已生成 kernel_launcher
    │
    ├─ cudagraphify_impl 包装：
    │     warmup 运行（正常执行一次，填充 static buffers）
    │     torch.cuda.graph(CUDAGraph()):
    │         kernel_launcher(static_inputs)
    │     ← 录制完成
    │
后续调用：
    ├─ copy_(new_inputs → static_inputs)  [CPU: 数据拷贝]
    └─ graph.replay()                      [CPU: 一条 Driver 命令]
                                           [GPU: 重放全部 kernel]
```

### 10.4 FSDP + torch.compile + CUDA Graph

```
                   分布式 4 GPU 场景（每 GPU 持有 1/4 参数）

model(x) on GPU 0
    │
    ├─ FSDP pre-hook [all-gather stream]:
    │     AllGather(weight_shard_0, weight_shard_1, weight_shard_2, weight_shard_3)
    │     → weight_full（完整参数，临时存在 GPU 0）
    │
    ├─ Dynamo 编译子图（使用 weight_full 计算）
    │     → Inductor kernel_launcher
    │     → CUDA Graph replay（在 compute stream 上）
    │     （与下一层的 all-gather 重叠）
    │
    ├─ FSDP post-hook [compute stream]:
    │     free(weight_full)  ← 释放完整参数，节省显存
    │
    └─ FSDP backward [reduce-scatter stream]:
          ReduceScatter(grad_full) → grad_shard_0（分片梯度归还给 GPU 0）
```

---

## 附录：关键文件索引

| 组件 | 核心文件 |
|---|---|
| fx.Tracer / Proxy | `torch/fx/_symbolic_trace.py`, `torch/fx/proxy.py` |
| fx.Graph / Node | `torch/fx/graph.py`, `torch/fx/node.py` |
| fx.GraphModule | `torch/fx/graph_module.py` |
| make_fx / ProxyTorchDispatchMode | `torch/fx/experimental/proxy_tensor.py` |
| Dynamo 入口 | `torch/_dynamo/eval_frame.py` |
| Dynamo 图构建 | `torch/_dynamo/output_graph.py` |
| AOTAutograd | `torch/_functorch/_aot_autograd/` |
| Inductor 入口 | `torch/_inductor/compile_fx.py` |
| CUDA Graph Trees | `torch/_inductor/cudagraph_trees.py` |
| FSDP2 主逻辑 | `torch/distributed/fsdp/_fully_shard/` |
| C++ Dispatcher Python 侧 | `torch/_ops.py` |
| FakeTensor | `torch/_subclasses/fake_tensor.py` |
| ShapeEnv | `torch/fx/experimental/symbolic_shapes.py` |
