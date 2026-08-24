---
title: PyTorch FX Graph 的三种构建途径：Dynamo、make_fx 与 Inductor fx_wrapper 深度对比
date: 2026-08-24 00:00:00 +0800
categories: [AI框架, PyTorch]
tags: [PyTorch, torch.compile, Inductor, Dynamo, FX, 编译器]
description: 系统分析 torch.compile 管线中三种 FX Graph 构建途径的源码级实现与设计哲学对比
---

# PyTorch FX Graph 的三种构建途径：Dynamo、make_fx 与 Inductor fx_wrapper 深度对比

> PyTorch 2.x 的编译流水线中，FX Graph 扮演着承上启下的角色--它是 Python 语义和底层编译器之间的桥梁。然而，FX Graph 并非由单一入口产生，而是在编译管线的不同阶段、通过截然不同的机制被构建。本文系统分析三种 FX Graph 构建途径：Dynamo 的字节码级符号执行、make_fx 的 dispatcher 级代理追踪、以及 Inductor fx_wrapper 的 IR 回写。通过源码级对比，揭示它们各自的设计哲学、能力边界与协作关系。

---

## 目录

1. [全景：三种构图途径在编译管线中的位置](#1-全景三种构图途径在编译管线中的位置)
2. [Dynamo：字节码级符号执行构图](#2-dynamo字节码级符号执行构图)
3. [make_fx：Dispatcher 级代理追踪构图](#3-make_fxdispatcher-级代理追踪构图)
4. [Inductor fx_wrapper：IR 回写构图](#4-inductor-fx_wrapperir-回写构图)
5. [三者对比：设计哲学、能力边界与数据流](#5-三者对比设计哲学能力边界与数据流)
6. [关键问题解答](#6-关键问题解答)
7. [总结](#7-总结)

---

## 1. 全景：三种构图途径在编译管线中的位置

在 torch.compile 的完整管线中，FX Graph 被构建了不止一次，而是**三次**，每次的构建者、目的和图语义都不同：

```
用户 Python 代码（含 if/for 等控制流）
  |
  v  1. Dynamo 构图（字节码级符号执行）
  |     InstructionTranslator 逐条解释 Python 字节码
  |     -> FX GraphModule #1（torch ops + guards）
  |
  v  2. make_fx 构图（dispatcher 级代理追踪）
  |     ProxyTorchDispatchMode 拦截 __torch_dispatch__
  |     -> FX GraphModule #2（joint fwd+bwd, ATen ops）
  |     -> selective_decompose 选择性算子分解
  |     -> FX GraphModule #3（分解后 ATen core ops）
  |
  v  3. Inductor fx_wrapper 构图（IR 回写为 FX）
  |     GraphLowering (FX Interpreter -> Inductor IR)
  |     -> Scheduler 融合 / 排序
  |     -> WrapperFxCodegen (IR -> FX GraphModule #4)
  |
  v  最终可执行 FX GraphModule（含 kernel 调用 + buffer 管理节点）
```

| 构建途径 | 构建者 | 拦截层级 | 产出图语义 | 典型调用方 |
|----------|--------|----------|-----------|-----------|
| **Dynamo** | InstructionTranslator | Python 字节码（PEP 523） | torch ops + guards | torch.compile 前端 |
| **make_fx** | ProxyTorchDispatchMode | Torch Dispatcher（__torch_dispatch__） | ATen ops（含 decomposition） | AOTAutograd, torch.export |
| **fx_wrapper** | WrapperFxCodegen | Inductor IR -> FX 回写 | 含 kernel 调用的 FX 图 | Inductor codegen 后端 |

![三种构图途径在编译管线中的位置](/assets/img/fx-graph-construction/diagram1-pipeline-overview.png){: width="100%" }

下面逐一深入分析。

---

## 2. Dynamo：字节码级符号执行构图

![Dynamo 字节码级符号执行构图机制](/assets/img/fx-graph-construction/diagram2-dynamo-mechanism.png){: width="100%" }

### 2.1 核心机制

Dynamo 的构图位于 `torch/_dynamo/symbolic_convert.py`，核心类是 `InstructionTranslator`。它通过 PEP 523 钩子接管 Python 帧执行，**逐条解释 Python 字节码**，将真实的 Python 执行转化为符号化的 FX 图构建。

```
用户调用 model(x)
  -> PEP 523 挂钩 eval_frame
  -> ConvertFrame.__call__ [convert_frame.py]
    -> InstructionTranslator 逐条遍历字节码指令
      LOAD_FAST         -> 从 symbolic_locals 取出 VariableTracker
      CALL_FUNCTION     -> 符号化调用，生成 FX call_function 节点
      POP_JUMP_IF_FALSE -> 生成 guard，记录当前路径
      不支持的指令       -> graph_break，分割为多个子图
    -> 输出 OutputGraph -> 多个 FX GraphModule + guards
```

### 2.2 关键源码分析

**符号化变量系统**：Dynamo 不执行真实计算，而是将每个 Python 值包装为 `VariableTracker` 子类：

```python
# symbolic_convert.py:1366
class InstructionTranslatorBase(metaclass=BytecodeDispatchTableMeta):
    output: OutputGraph               # 正在构建的 FX 图
    symbolic_locals: dict[str, VariableTracker]  # 符号化局部变量
    symbolic_globals: dict[str, VariableTracker]  # 符号化全局变量
    stack: list[VariableTracker]      # 模拟的值栈
    block_stack: list[BlockStackEntry]  # 控制流块栈
    instruction_pointer: int | None   # 当前字节码位置
```

Dynamo 维护了一个完整的"虚拟 Python"——值栈、局部变量表、控制流块栈都是符号化的。当遇到 `CALL_FUNCTION` 字节码时，它不会真正调用函数，而是通过 `VariableTracker.call_function()` 在 FX 图中创建一个 `call_function` 节点。

**控制流处理**：当遇到 `POP_JUMP_IF_FALSE` 等条件跳转指令时，Dynamo 生成 guard 并只追踪一条路径：

```python
# 简化的条件分支处理逻辑
def POP_JUMP_IF_FALSE(self, inst):
    # 尝试将条件值常量折叠
    if self.can_resolve_condition(inst):
        # 静态可判断 -> 直接选择路径，不生成 guard
        ...
    else:
        # 生成 guard，运行时检查条件
        guard = GuardBuilder(inst, self)
        install_guard(guard)
        # 只追踪当前路径（true 或 false），另一条路径被"放弃"
```

这意味着 Dynamo **在图构建时就已经消除了控制流**——图只包含一条执行路径，guard 保证运行时路径一致，否则触发重新编译。

> **关键区分：形状依赖 vs 数据依赖控制流**
>
> Dynamo 的 guard 机制只能处理**形状依赖**控制流（如 `if x.shape[0] > 10`）：shape 被特化为常量，条件在编译时静态求值，guard 检查的是 tensor 的 shape/dtype 等元信息。
>
> 对于**数据依赖**控制流（如 `if x.sum() > 0`），条件值取决于 tensor 数据，无法在编译时静态求值。Dynamo **不会生成 guard**，而是触发 **graph break**：第一个子图计算条件值并返回，Python `if` 在 eager 中执行，第二个子图追踪选定分支。真实运行验证（`if x.sum() > 0`）产生 2 个子图——子图 #1 只含 `sum → gt`，子图 #2 含 relu 分支，中间 `if` 不在任何图中。

**Graph Break 机制**：遇到不支持的指令时，Dynamo 不是报错，而是优雅地分割图：

```python
# 简化的 graph_break 流程
def graph_break(self, reason):
    # 1. 关闭当前子图
    self.output.compile_subgraph(self)
    # 2. 生成 resume 函数，用于从 break 点恢复执行
    resume_fn = create_resume_fn(self)
    # 3. 在 break 点插入 eager 回退
    # 4. 新建 InstructionTranslator 继续
```

### 2.3 产出图的特性

Dynamo 产出的 FX Graph（GraphModule #1）具有以下特征：

- **节点 target 是 OpOverload**：如 `torch.ops.aten.add.Tensor`，而非高级 Python API
- **控制流已消除**：图是线性的 op 序列，无 if/for 节点
- **可能包含 HigherOrderOperator**：如 `torch.cond`、`torch.while_loop` 会作为图中的 `call_function` 节点保留
- **附带 guards**：每个图都有一组运行时检查条件
- **可能不完整**：graph break 导致多个子图，子图间回退 eager

---

## 3. make_fx：Dispatcher 级代理追踪构图

![make_fx Dispatcher 级代理追踪构图机制](/assets/img/fx-graph-construction/diagram3-makefx-mechanism.png){: width="100%" }

### 3.1 核心机制

`make_fx` 位于 `torch/fx/experimental/proxy_tensor.py`，核心是通过 `ProxyTorchDispatchMode` 拦截 Torch Dispatcher 层的 `__torch_dispatch__` 调用。与 Dynamo 的"模拟执行"不同，make_fx **实际执行**用户函数，但在执行过程中拦截每个 op，为其创建对应的 FX 节点。

```
make_fx(f)(args)
  -> _MakefxTracer.trace(f, *args)
    -> _init_modes_from_inputs: 构建模式栈
      |- PythonKeyTracer (FX 图构建器)
      |- FakeTensorMode (可选, "fake"/"symbolic")
      +- ProxyTorchDispatchMode (核心拦截器)
    -> _trace_inner:
      -> wrap_key(func, args, fx_tracer): 建立 tensor->proxy 映射
      -> dispatch_trace(wrap_key(...), tracer=fx_tracer)
        -> 实际执行 func(*args)
        -> 每个 torch op -> __torch_dispatch__ -> proxy_call()
```

### 3.2 关键源码分析

**proxy_call：拦截与节点创建的核心**（`proxy_tensor.py:1269`）：

```python
def proxy_call(proxy_mode, func, pre_dispatch, args, kwargs):
    # 1. 尝试 decomposition（如有分解表）
    r = maybe_handle_decomp(proxy_mode, func, args, kwargs)
    if r is not NotImplemented:
        return r

    # 2. 获取每个参数对应的 Proxy
    f_flat_args_kwargs, proxy_flat_args_kwargs, all_constant = (
        _fetch_proxies_and_all_constant_flag(flat_args_kwargs, tracer)
    )

    # 3. 数据依赖 op 的特殊处理
    if torch.Tag.data_dependent_output in func.tags:
        if all_constant:
            return func(*const_args, **const_kwargs)
        if proxy_mode._error_on_data_dependent_ops:
            raise RuntimeError(
                "It appears that you're trying to get value out of a "
                "tracing tensor - erroring out! It may be possible to "
                "trace this with dynamic shapes; try tracing_mode='symbolic'"
            )

    # 4. 在 FX 图中创建 call_function 节点
    proxy_out = proxy_mode.tracer.create_proxy(
        "call_function", func, proxy_args, proxy_kwargs)

    # 5. 实际执行 op（在 FakeTensor 模式下不产生真实计算）
    out = func(*args, **kwargs)

    # 6. 将输出 tensor 与新建的 proxy 关联
    track_tensor_tree(out, proxy_out, constant=constant, tracer=tracer)
    return out
```

核心逻辑是：每次 torch op 经过 C++ dispatcher 时，`__torch_dispatch__` 被触发 -> 创建一个 FX Node -> 执行 op -> 将输出 tensor 与 proxy 绑定。后续操作使用这个 tensor 时，自动找到对应的 proxy，从而串成完整的计算图。

**三种追踪模式**：

| 模式 | 行为 | FakeTensor | ShapeEnv | 适用场景 |
|------|------|------------|----------|---------|
| `"real"` | 真实 tensor 执行 | 否 | 否 | 调试、简单场景 |
| `"fake"` | FakeTensor 执行，维度静态 | 是 | 是(可选) | AOT 编译、export |
| `"symbolic"` | FakeTensor + 所有维度为 backed SymInt | 是 | 是 | 符号化形状推理 |

**AOTAutograd 中的两步调用**：

第一步——初始追踪（`aot_autograd.py:1737`），注意**不传 decomposition_table**：

```python
fx_g = make_fx(flattened_joint, record_module_stack=True)(*full_args)
```

此时所有 op 被原样记录，与输入图的 op 级别一致。

第二步——选择性分解（`proxy_tensor.py:2412-2437`）：

```python
def selective_decompose(joint_gm, *args, decomposition, should_decompose, ...):
    def wrap_fn(*args):
        return _SelectiveDecomposeInterpreter.recursive_wrap(
            joint_gm, should_decompose, decomposition
        ).run(*args)
    return make_fx(wrap_fn, decomposition_table={})(*args)  # 第二次 make_fx
```

`_SelectiveDecomposeInterpreter` 继承 `fx.Interpreter`，逐节点遍历 GraphModule，对需要分解的节点临时启用 decomposition table，不需要分解的节点原样执行。这个过程**递归处理** HigherOrderOperator 的子图——通过 `recursive_wrap` 方法，HOP 参数中的子 GraphModule 也被包装，实现子图内的选择性分解。

### 3.3 产出图的特性

`make_fx` 产出的 FX Graph（GraphModule #2/#3）具有以下特征：

- **精确的执行序列**：记录的就是实际执行的 op 序列，无"猜测"
- **无控制流**：Python 控制流已被实际执行，不会出现在图中。直接使用 make_fx 追踪含控制流的函数存在**静默错误分支**隐患——追踪时只捕获一条分支且不生成 guard，换输入运行时若应走另一分支则结果错误。
- **不完全保护**：make_fx 对**数据依赖**条件（`if x.sum() > 0`）有保护：`if <tensor>:` 隐式调用 `aten._local_scalar_dense.default`（带 `data_dependent_output` tag），过 dispatch 时被拦截报错。但对**形状依赖**条件（`if x.shape[0] > 10` 或 `if x.numel() > 10`），条件全程在 Python 层求值（`shape`/`numel` 返回 Python int，不过 dispatch），make_fx 完全看不到，**静默错误分支**。这也是标准管线中 Dynamo 必须在 make_fx 之前的原因——Dynamo 在字节码层能看到所有控制流
- **支持 decomposition**：通过 `decomposition_table` 将复合 op 分解为底层 ATen op
- **附带完整 metadata**：每个节点都有 `tensor_meta`（shape/dtype/stride）和 `val`
- **可包含 HOP 子图**：`torch.cond` 等 HigherOrderOperator 会递归追踪子图

---

## 4. Inductor fx_wrapper：IR 回写构图

![Inductor fx_wrapper IR 回写构图机制](/assets/img/fx-graph-construction/diagram4-fxwrapper-mechanism.png){: width="100%" }

### 4.1 核心机制

fx_wrapper 是 Inductor 的**第三种 codegen 后端**（Python wrapper / C++ wrapper / FX wrapper），通过 `config.fx_wrapper = True` 启用。与前两种构图的"从 Python 代码捕获图"不同，fx_wrapper 是**从 Inductor 的优化后 IR 反向生成 FX 图**——编译产物本身是一个 FX GraphModule。

```
FX Graph (post-grad, ATen ops)
  -> GraphLowering (torch.fx.Interpreter 子类)
    -> 逐节点遍历 -> 分派到 lowering 函数 -> 生成 Inductor IR
    -> Scheduler: 融合、排序、buffer 管理
    -> WrapperFxCodegen: 将 IR 操作序列转换回 FX 节点
      -> FxConverter: 逐条 Wrapper IR Line -> FX call_function 节点
    -> 输出: FX GraphModule (含 Triton kernel 调用节点)
```

### 4.2 关键源码分析

**GraphLowering 作为 FX Interpreter**（`graph.py:365`）：

```python
class GraphLowering(torch.fx.Interpreter):
    """Lowers an FX graph to Inductor IR and drives backend code generation.

    Walks the FX graph node-by-node, materializing inputs/outputs and
    constants, dispatching each call to an Inductor lowering, and accumulating
    the resulting IR nodes.
    """

    def __init__(self, gm, example_inputs=None, shape_env=None,
                 cpp_wrapper=False, fx_wrapper=False, ...):
        super().__init__(gm)       # 继承 FX Interpreter
        self.fx_wrapper = fx_wrapper
        self.operations: list[ir.Operation] = []  # IR 节点累积器
```

GraphLowering 遍历输入的 FX 图，对每个 `call_function` 节点查找对应的 lowering：

```python
# graph.py 中的 lowering 查找链 (简化)
def call_function(self, target, args, kwargs):
    if target in user_lowerings:          # 用户自定义 lowering (最高优先级)
        out = user_lowerings[target](*args, **kwargs)
    elif target in lowerings:             # 内置 lowering
        out = lowerings[target](*args, **kwargs)
    else:                                 # 兜底: FallbackKernel
        out = fallback_handler(target)(*args, **kwargs)
    # out 是 TensorBox/StorageBox/Pointwise 等 IR 节点
    self.operations.append(...)
    return out
```

**WrapperFxCodegen：IR -> FX 图的回写**（`wrapper_fxir.py:43`）：

```python
class WrapperFxCodegen(PythonWrapperCodegen):
    """Backend to generate wrapper code as an FX IR graph."""

    def _generate(self, is_inference):
        self.run_wrapper_ir_passes(is_inference)
        # 将 IR 操作序列转换为 FX 节点
        gm = FxConverter(
            lines=self.lines,               # Wrapper IR Line 列表
            prologue=prologue,
            graph_inputs=self.get_fx_graph_inputs(),
            graph_outputs=self.get_graph_outputs(),
            subgms=self.subgms,             # 子图 GraphModule
            is_subgraph=self.is_subgraph,
        ).generate()
        compiled_fn = self.compile_graph(gm)
        return FileBackedGraphModule(gm, compiled_fn), None
```

**FxConverter：逐行转换**（`wrapper_fxir.py:182`）：

`FxConverter` 是一个 dataclass，接收 Wrapper IR 的 `Line` 列表，逐条转换为 FX 节点。每条 `KernelCallLine` 变为一个 `call_function` 节点，target 是 `triton_kernel_wrapper_mutation`（Triton 融合 kernel）或 ATen fallback op：

```python
@dataclasses.dataclass
class FxConverter:
    lines: list[Line]           # Wrapper IR 操作序列
    graph_inputs: dict[...]     # 图输入
    graph_outputs: list[...]    # 图输出
    subgms: dict[str, GraphModule]  # 子图

    def generate(self) -> GraphModule:
        graph = torch.fx.Graph()
        # 遍历每条 Line，创建对应的 FX 节点
        for line in self.lines:
            self._convert_line(line, graph)
        return GraphModule({}, graph)
```

**三种 codegen 后端的选择**（`common.py:483`）：

```python
def get_wrapper_codegen_for_device(device, cpp_wrapper=False, fx_wrapper=False):
    if fx_wrapper:
        return wrapper_codegen_obj.fx_wrapper_codegen   # FX IR 路径
    elif cpp_wrapper:
        return wrapper_codegen_obj.cpp_wrapper_codegen   # C++ 路径
    else:
        return wrapper_codegen_obj.wrapper_codegen       # Python 路径
```

### 4.3 产出图的特性

fx_wrapper 产出的 FX Graph（GraphModule #4）具有以下特征：

- **节点是 kernel 调用**：`call_function` 的 target 是 `triton_kernel_wrapper_mutation`（Triton 融合 kernel）或 extern kernel，而非原始 ATen op
- **包含融合后的 op**：多个原始 op 被融合为单个 kernel 调用节点
- **保留 buffer 管理**：包含 buffer 分配、复用、释放节点
- **可包含子图**：conditional / subgraph 作为嵌套 GraphModule
- **可被 FX 工具链处理**：因为是标准 FX GraphModule，可以被 FX passes 进一步优化或序列化

---

## 5. 三者对比：设计哲学、能力边界与数据流

### 5.1 完整对比表

| 维度 | Dynamo | make_fx | fx_wrapper |
|------|--------|---------|------------|
| **源码位置** | `torch/_dynamo/symbolic_convert.py` | `torch/fx/experimental/proxy_tensor.py` | `torch/_inductor/codegen/wrapper_fxir.py` |
| **核心类** | `InstructionTranslator` | `ProxyTorchDispatchMode` / `_MakefxTracer` | `GraphLowering` / `WrapperFxCodegen` |
| **拦截机制** | PEP 523 字节码钩子 | `__torch_dispatch__` | FX Interpreter + IR 回写 |
| **执行方式** | 符号化模拟（不真正执行） | 实际执行（FakeTensor 下） | IR 遍历（无执行） |
| **拦截层级** | Python 字节码（最上层） | Torch Dispatcher（中间层） | Inductor IR（最底层） |
| **设计目标** | 最大化 Python 语义覆盖 | 精确捕获 op 执行序列 | 产出可序列化/可分析的编译产物 |
| **图的角色** | 输入：用户代码 -> 图 | 输入：图 -> 图（重追踪） | 输出：IR -> 图（回写） |
| **Python 控制流** | 支持（通过 guard 消除） | 不支持（直接执行） | N/A（输入已无控制流） |
| **Graph Break** | 支持（优雅降级） | 不支持（报错） | N/A |
| **自动重编译** | 支持（guard 失败时） | 不支持 | N/A |
| **算子分解** | 不支持 | 支持（decomposition_table） | N/A（在 lowering 阶段处理） |
| **前向+反向联合追踪** | 不支持 | 支持（AOTAutograd 联合） | N/A |
| **kernel 融合** | 不支持 | 不支持 | 支持（Scheduler 融合后回写） |
| **buffer 管理** | 不支持 | 不支持 | 支持（含分配/复用/释放节点） |
| **HOP 子图递归** | 支持（inline 子图） | 支持（recursive_wrap） | 支持（subgms 嵌套） |
| **动态形状** | 支持（guard + SymInt） | 支持（symbolic 模式） | 支持（sizevars） |
| **典型调用方** | torch.compile 前端 | AOTAutograd / torch.export | Inductor codegen |
| **图语义层次** | torch ops + guards | ATen core ops | kernel 调用 + buffer 管理 |

### 5.2 数据流：从用户代码到最终产物

下面用一个具体例子说明三种构图途径如何接力工作。以下所有 FX Graph 均为**真实运行 dump**，运行环境为 PyTorch main (2025)、CPU 后端。用户代码为：

```python
@torch.compile
def model(x, w):
    if x.shape[0] > 10:       # 形状依赖控制流
        y = torch.relu(x @ w)
    else:
        y = torch.nn.functional.gelu(x @ w)
    return y * 2
```

输入：`x = torch.randn(16, 8, requires_grad=True)`，`w = torch.randn(8, 4, requires_grad=True)`，满足 `x.shape[0]=16 > 10`，追踪 relu 分支。

> **形状依赖 vs 数据依赖控制流**
>
> Dynamo 的 guard 机制只能处理**形状依赖**控制流（如 `if x.shape[0] > 10`）：shape 被特化为常量，条件在编译时静态求值，guard 检查的是 tensor 的 shape/dtype 等元信息。
>
> 对于**数据依赖**控制流（如 `if x.sum() > 0`），条件值取决于 tensor 数据，无法在编译时静态求值。Dynamo **不会生成 guard**，而是触发 **graph break**：第一个子图计算条件值并返回，Python `if` 在 eager 中执行，第二个子图追踪选定分支。真实运行验证（`if x.sum() > 0`）产生 2 个子图——子图 #1 只含 `sum → gt`，子图 #2 含 relu 分支，中间 `if` 不在任何图中。

**第一阶段：Dynamo 构图**

Dynamo 遇到 `if x.shape[0] > 10` 时，将 `x.shape[0]` 特化为常量 16，在编译时静态求值为 `True`，只追踪 true 分支（relu 路径）。产出的 GraphModule #1（`gm.print_readable()` 真实输出）：

```python
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[16, 8]", L_w_: "f32[8, 4]"):
        l_x_ = L_x_
        l_w_ = L_w_

        # code: y = torch.relu(x @ w)
        matmul: "f32[16, 4]" = l_x_ @ l_w_;  l_x_ = l_w_ = None
        y: "f32[16, 4]" = torch.relu(matmul);  matmul = None

        # code: return y * 2
        mul: "f32[16, 4]" = y * 2;  y = None
        return (mul,)
```

关键特征：
- **控制流已消除**：图中只有 relu 路径，无 if/else 节点
- **op 是高级 Python API**：`l_x_ @ l_w_`、`torch.relu`、`y * 2`（尚未分解为 ATen op）
- **附带 guards**：包括 `TENSOR_MATCH`（检查 x/w 的 shape/dtype/stride）、`SHAPE_ENV`、`GRAD_MODE` 等。当 x.shape 从 (16, 8) 变为 (5, 8) 时，guard 失败，Dynamo 重编译并追踪 gelu 分支：

```python
# 重编译后的 gelu 分支图（x.shape[0]=5 <= 10，真实 dump）
class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", L_x_: "f32[s77, 8]", L_w_: "f32[8, 4]"):
        l_x_ = L_x_
        l_w_ = L_w_

        matmul: "f32[s77, 4]" = l_x_ @ l_w_;  l_x_ = l_w_ = None
        y: "f32[s77, 4]" = torch._C._nn.gelu(matmul);  matmul = None
        mul: "f32[s77, 4]" = y * 2;  y = None
        return (mul,)
```

注意第二次编译将 `x.shape[0]` 标记为符号 `Sym(s77)`（动态形状），避免后续再次重编译。

**第二阶段：make_fx 构图**

AOTAutograd 将 Dynamo 产出的 GraphModule #1 包装为 joint function（前向+反向），调用 `make_fx` 追踪。make_fx 第一次调用**不传 decomposition_table**，所有 op 被原样记录。产出的 GraphModule #2（真实 dump）：

```python
class joint_fn(torch.nn.Module):
    def forward(self, x_1: "f32[16, 8]", w_1: "f32[8, 4]", tangent_1: "f32[16, 4]"):
        # 前向
        mm: "f32[16, 4]" = torch.ops.aten.mm.default(x_1, w_1)
        relu: "f32[16, 4]" = torch.ops.aten.relu.default(mm);  mm = None
        detach: "f32[16, 4]" = torch.ops.aten.detach.default(relu);  detach = None
        mul: "f32[16, 4]" = torch.ops.aten.mul.Tensor(relu, 2)
        # 反向
        mul_1: "f32[16, 4]" = torch.ops.aten.mul.Tensor(tangent_1, 2);  tangent_1 = None
        threshold_backward: "f32[16, 4]" = torch.ops.aten.threshold_backward.default(
            mul_1, relu, 0);  mul_1 = relu = None
        t: "f32[8, 16]" = torch.ops.aten.t.default(x_1);  x_1 = None
        mm_1: "f32[8, 4]" = torch.ops.aten.mm.default(t, threshold_backward);  t = None
        t_1: "f32[4, 8]" = torch.ops.aten.t.default(w_1);  w_1 = None
        mm_2: "f32[16, 8]" = torch.ops.aten.mm.default(threshold_backward, t_1)
        return (mul, mm_2, mm_1)
```

关键观察：
- **Python API 已分解为 ATen op**：`l_x_ @ l_w_` → `torch.ops.aten.mm.default`，`torch.relu` → `torch.ops.aten.relu.default`
- **`detach` 节点出现**：autograd 在 relu 和 mul 之间插入 `detach`，用于保存前向输出供反向使用
- **反向已是 `threshold_backward`**（而非 `relu_backward`）：`torch.relu` 的反向在 autograd 引擎内已被默认分解为 `threshold_backward`
- **joint 图线性排列**：前向和反向在同一图中，无控制流

随后 `selective_decompose` 将部分复合 op 进一步分解为底层 ATen op。产出 GraphModule #3（真实 dump）：

```python
class joint_fn(torch.nn.Module):
    def forward(self, x_1: "f32[16, 8]", w_1: "f32[8, 4]", tangent_1: "f32[16, 4]"):
        # 前向
        mm: "f32[16, 4]" = torch.ops.aten.mm.default(x_1, w_1)
        relu: "f32[16, 4]" = torch.ops.aten.relu.default(mm);  mm = None
        alias: "f32[16, 4]" = torch.ops.aten.alias.default(relu);  alias = None
        mul: "f32[16, 4]" = torch.ops.aten.mul.Tensor(relu, 2)
        # 反向
        mul_1: "f32[16, 4]" = torch.ops.aten.mul.Tensor(tangent_1, 2);  tangent_1 = None
        le: "b8[16, 4]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, ...)
        where: "f32[16, 4]" = torch.ops.aten.where.self(le, scalar_tensor, mul_1)
        permute: "f32[8, 16]" = torch.ops.aten.permute.default(x_1, [1, 0]);  x_1 = None
        mm_1: "f32[8, 4]" = torch.ops.aten.mm.default(permute, where);  permute = None
        permute_1: "f32[4, 8]" = torch.ops.aten.permute.default(w_1, [1, 0]);  w_1 = None
        mm_2: "f32[16, 8]" = torch.ops.aten.mm.default(where, permute_1)
        return (mul, mm_2, mm_1)
```

`selective_decompose` 产生的变化（对比 #2 → #3）：

| 原始 op (#2) | 分解后 op (#3) | 说明 |
|---|---|---|
| `threshold_backward` | `le` + `scalar_tensor` + `where` | relu 反向 = `where(x<=0, 0, grad)` |
| `detach` | `alias` | detach 语义等价于 alias |
| `t.default` | `permute.default(..., [1, 0])` | 转置用 permute 表示 |

**第三阶段：Inductor fx_wrapper 构图**

GraphLowering 遍历 GraphModule #3，将每个 ATen op 转换为 Inductor IR。Scheduler 融合相邻的 pointwise op。WrapperFxCodegen 将融合后的 IR 回写为 GraphModule #4（真实 dump，CPU 后端，`config.fx_wrapper=True`）：

> **注意：CPU 融合 kernel 的支持状态**
>
> PyTorch 官方版本的 fx_wrapper 仅支持 Triton 后端，产出 `triton_kernel_wrapper_mutation` 节点。以下 CPU 后端的 `compiled_kernel_wrapper_mutation`（C++ 融合 kernel）示例基于一个讨论中的 PR：[pytorch/pytorch#187938](https://github.com/pytorch/pytorch/pull/187938)。在该 PR 合入前，官方版本在 CPU 上不会产生融合 kernel 节点，所有 op 以 `aten.*.out` fallback 形式出现。

前向图：

```python
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 8]", primals_2: "f32[8, 4]"):
        buf0: "f32[16, 4]" = torch.empty_strided([16, 4], [4, 1], ...)
        mm_out: "f32[16, 4]" = torch.ops.aten.mm.out(
            primals_1, primals_2, out = buf0);  mm_out = None
        buf1: "f32[16, 4]" = torch.empty_strided([16, 4], [4, 1], ...)
        buf2: "b8[16, 4]" = torch.empty_strided([16, 4], [4, 1], dtype = torch.bool, ...)
        compiled_kernel_wrapper_mutation = torch.ops.higher_order.compiled_kernel_wrapper_mutation(
            kernel_idx = 0, mutated_arg_indices = (1, 2), args = (buf0, buf1, buf2))
        primals_1_view: "f32[8, 16]" = torch.as_strided(primals_1, [8, 16], [1, 8], 0)
        primals_2_view: "f32[4, 8]" = torch.as_strided(primals_2, [4, 8], [1, 4], 0)
        return [buf1, buf2, primals_1_view, primals_2_view]
```

反向图：

```python
class GraphModule(torch.nn.Module):
    def forward(self, le: "b8[16, 4]", permute: "f32[8, 16]",
               permute_1: "f32[4, 8]", tangents_1: "f32[16, 4]"):
        buf0: "f32[16, 4]" = torch.empty_strided([16, 4], [4, 1], ...)
        compiled_kernel_wrapper_mutation = torch.ops.higher_order.compiled_kernel_wrapper_mutation(
            kernel_idx = 1, mutated_arg_indices = (2,), args = (le, tangents_1, buf0))
        buf1: "f32[8, 4]" = torch.empty_strided([8, 4], [4, 1], ...)
        mm_out: "f32[8, 4]" = torch.ops.aten.mm.out(permute, buf0, out = buf1)
        buf2: "f32[16, 8]" = torch.empty_strided([16, 8], [8, 1], ...)
        mm_out_1: "f32[16, 8]" = torch.ops.aten.mm.out(buf0, permute_1, out = buf2)
        return [buf2, buf1]
```

关键特征：
- **节点是 kernel 调用**：融合后的 kernel（融合了 relu + mul + le + where 等 pointwise op）。CPU 后端（上述示例）为 `compiled_kernel_wrapper_mutation`；CUDA/Triton 后端为 `triton_kernel_wrapper_mutation`
- **buffer 管理显式化**：`torch.empty_strided` 分配、`out=` 参数复用 buffer
- **`mm.out` 替代 `mm.default`**：使用 out-variant 避免临时分配
- **`as_strided` 替代 `permute`**：转置用零拷贝 view 实现（IR 层面进一步优化）
- **前向保存中间值**：`buf1`（mul 输出）、`buf2`（le 输出）、`primals_1_view`/`primals_2_view`（转置 view）供反向使用
- 融合 kernel 节点名为 `triton_kernel_wrapper_mutation(kernel_idx=N, ...)`（Triton 后端）或 `compiled_kernel_wrapper_mutation(kernel_idx=N, ...)`（CPU 后端），均不包含 `mm`（GEMM 不与 pointwise 融合，仍为单独 `mm.out` 调用）

最终产物是一个 FX GraphModule，其节点是 kernel 调用——既可执行，又可被 FX 工具链进一步分析或序列化。

### 5.3 图的"语义层次"演进

三种构图途径产出的图，在语义层次上逐步降低：

```
Dynamo 图 (#1)     : torch ops + guards     <- 最接近用户语义 (Python API: @, relu, *)
    |
    v  make_fx 重追踪 + decomposition
make_fx 图 (#2/#3)  : ATen core ops          <- 精确的算子序列 (mm.default, threshold_backward)
    |
    v  GraphLowering + Scheduler + WrapperFxCodegen
fx_wrapper 图 (#4) : kernel 调用 + buffer 管理  <- 最接近硬件执行 (triton_kernel_wrapper_mutation, mm.out)
```

每一次转换都**丢失了一些信息**，但也**获得了一些能力**：
- Dynamo -> make_fx：丢失了 guard 信息，获得了 decomposition 和 joint 追踪
- make_fx -> fx_wrapper：丢失了原始 op 粒度，获得了 kernel 融合和 buffer 管理

---

## 6. 关键问题解答

### Q1: Dynamo 成图后经过 make_fx，会被记录成原始的图还是拆成内部算子？

**两步走——先原样记录，再选择性分解。**

第一步（`aot_autograd.py:1737`）：`make_fx` 调用**不传 decomposition_table**，所有 op 被原样记录。此时 Python API（`@`, `torch.relu`）已通过 dispatcher 层自然映射为对应的 ATen op（`aten.mm.default`, `aten.relu.default`），但**不做进一步分解**——如 `threshold_backward` 保持原样，不会被拆成 `le + where`。

第二步（`proxy_tensor.py:2412`）：`selective_decompose` 使用 `_SelectiveDecomposeInterpreter` 逐节点遍历，对需要分解的节点临时启用 decomposition table，不需要分解的节点原样保留。最终产出的是**部分分解**的图。真实示例中，`threshold_backward` 被分解为 `le + scalar_tensor + where`，`detach` 被分解为 `alias`，`t.default` 被分解为 `permute.default`，而 `mm.default` 和 `relu.default` 保持原样。

### Q2: 其中的控制流是否会被消除？

**形状依赖控制流会在 Dynamo 阶段消除（通过 guard）；数据依赖控制流则触发 graph break，不在任何图中保留。**

Dynamo 通过 guard 机制，在图构建时只追踪一条控制流路径（**仅限形状依赖条件**，如 `if x.shape[0] > 10`）。当 `make_fx` 接收到 Dynamo 产出的 FX 图时，图中已经没有任何 Python 控制流——它是线性的 op 序列。`make_fx` 重新追踪这个线性序列，产出的新图自然也不含控制流。

对于**数据依赖**控制流（如 `if x.sum() > 0`），Dynamo 会 graph break：条件计算和分支体被分割到不同子图中，Python `if` 在 eager 中执行，不在任何 FX 图中出现。真实运行验证：`if x.sum() > 0` 产生 2 个子图，子图 #1 只含 `sum → gt` 返回布尔值，子图 #2 含选定分支的 op 序列。

> **make_fx 直接追踪时的隐患**：若不经 Dynamo 直接用 make_fx 追踪含控制流的函数，结果图无 guard，存在静默错误分支风险。make_fx 对数据依赖条件有间接保护（`if <tensor>:` → `Tensor.__bool__` → `aten._local_scalar_dense` 带 `data_dependent_output` tag → dispatch 拦截报错），但对形状依赖条件（`x.shape[0]`、`x.numel()` 返回 Python int，不过 dispatch）无保护，会静默捕获错误分支。

唯一例外是 `torch.cond` / `torch.while_loop` 等 HigherOrderOperator——它们在图中作为 `call_function` 节点保留，`make_fx` 会递归追踪其子图，但子图内部同样不含 Python 控制流。

### Q3: fx_wrapper 产出的图与前两种图有什么本质区别？

**前两种图是"输入图"——描述用户要做什么；fx_wrapper 的图是"输出图"——描述编译器怎么做。**

Dynamo 和 make_fx 产出的图，节点是 ATen op，描述的是计算语义（如"做矩阵乘法"）。fx_wrapper 产出的图，节点是 kernel 调用，描述的是执行方案（如"调用 `triton_kernel_wrapper_mutation(kernel_idx=0)` 融合 kernel"）。

这意味着 fx_wrapper 的图：
- 不可跨硬件移植（kernel 是设备特定的）
- 不可进一步 decomposition（kernel 已是最终形态）
- 但可以直接执行（无需再经过编译）
- 且可被 FX 工具链分析（如可视化、序列化、AOT 部署）

### Q4: 三种构图途径是否可以独立使用？

**可以，但通常它们是接力关系。**

- **Dynamo 独立使用**：`torch.compile(backend="eager")` 只用 Dynamo 捕获图，不做后续编译
- **make_fx 独立使用**：`torch.export` 内部直接调用 make_fx（不经过 Dynamo），用于 AOT 导出。此时需注意：make_fx 在 dispatcher 层拦截，看不到 Python 层的形状依赖控制流（`x.shape[0]`、`x.numel()` 返回 Python int 不过 dispatch），追踪结果无 guard，存在静默错误分支风险。数据依赖控制流则因 `if <tensor>:` 隐式触发 `aten._local_scalar_dense`（带 `data_dependent_output` tag）会被拦截报错。`torch.export` 通过要求输入满足约束（`dynamic_shapes`、`eq` 约束等）来规避此问题
- **fx_wrapper 独立使用**：需要已有 Inductor IR，通常不脱离 Inductor 管线

在标准 `torch.compile(backend="inductor")` 管线中，三者按 Dynamo -> make_fx -> fx_wrapper 的顺序接力工作。

---

## 7. 总结

### 设计哲学总结

1. **Dynamo 解决的是"Python 语义覆盖"问题**——如何将灵活的 Python 代码（控制流、闭包、生成器等）转化为可编译的图。它的核心创新是字节码级符号执行 + guard 系统，使得用户无需修改代码即可获得编译加速。

2. **make_fx 解决的是"精确 op 捕获"问题**——如何确保图中记录的 op 序列与实际执行完全一致。它的核心创新是 dispatcher 层拦截，使得每个经过 PyTorch C++ 运行时的 op 都被精确记录，包括 decomposition 和 joint 追踪。

3. **fx_wrapper 解决的是"编译产物可分析"问题**——如何让 Inductor 的编译产出不仅是可执行的代码字符串，还是一个结构化的 FX 图。它的核心创新是 IR -> FX 的反向回写，使得编译产物可以被 FX 工具链进一步处理、序列化或部署。

三者接力工作，构成了从用户 Python 代码到高效 kernel 调用的完整转换链：**Dynamo 捕获语义 -> make_fx 精确化算子 -> fx_wrapper 回写执行方案**。每一步都在前一步的基础上进一步降低抽象层次，最终产出可直接执行的 kernel 调用图。

---

*参考代码版本：PyTorch main branch（2025）*
*核心文件：*
- *`torch/_dynamo/symbolic_convert.py` - Dynamo 字节码符号执行*
- *`torch/_dynamo/convert_frame.py` - Dynamo 帧转换入口*
- *`torch/fx/experimental/proxy_tensor.py` - make_fx 追踪核心*
- *`torch/_functorch/aot_autograd.py` - AOTAutograd 调用 make_fx*
- *`torch/_inductor/graph.py` - GraphLowering (FX Interpreter)*
- *`torch/_inductor/codegen/wrapper_fxir.py` - WrapperFxCodegen (IR -> FX)*
- *`torch/_inductor/codegen/common.py` - 三种 codegen 后端分发*
