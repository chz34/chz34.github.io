# AI Agent 执行全解：从用户意图到一次 LLM API 调用

> 当我们说"让 AI 帮我完成一个复杂任务"，背后发生了什么？本文从执行层级的视角，完整拆解 AI Agent 的运行机制——包括每一层的职责、嵌套关系、以及那个被反复讨论却鲜少说清的问题：**什么时候才真的需要自己构建 Agent？**

---

## 一、什么是 Agent Workflow

在讨论执行层级之前，先对齐一个基础认知：所谓"Agent"，本质上是一个**围绕 LLM 调用构建的控制循环**。LLM 本身是无状态的，每次调用它只看当前的输入文本，不记得任何历史。Agent 框架的全部工作，就是维护这个上下文，决定什么时候调用 LLM、怎么组织输入、怎么处理输出。

常见的 workflow 模式包括：**链式**（顺序执行）、**并行**（同时执行多个分支）、**路由**（按意图分发）、**ReAct**（推理-行动循环）、**Plan-and-Execute**（先规划再执行）、**多 Agent 协作**、以及**反思/自我修正**。

这些模式不是互斥的，实际系统往往是它们的嵌套组合。

---

## 二、Agent 执行栈：七个层级

整个执行过程可以分为七层，从最高的业务抽象一直到最底层的 HTTP 调用：

<svg width="100%" viewBox="0 0 680 520" xmlns="http://www.w3.org/2000/svg">
<defs>
  <style>
    .th { font-family: sans-serif; font-size: 14px; font-weight: 500; fill: #1a1a1a; }
    .ts { font-family: sans-serif; font-size: 12px; fill: #555; }
    .layer-coral { fill: #FAECE7; stroke: #993C1D; }
    .layer-purple { fill: #EEEDFE; stroke: #534AB7; }
    .layer-blue { fill: #E6F1FB; stroke: #185FA5; }
    .layer-amber { fill: #FAEEDA; stroke: #854F0B; }
    .layer-teal { fill: #E1F5EE; stroke: #0F6E56; }
    .layer-green { fill: #EAF3DE; stroke: #3B6D11; }
    .layer-gray { fill: #F1EFE8; stroke: #5F5E5A; }
    .arr { stroke: #888; stroke-width: 1.5; fill: none; }
    .label-r { font-family: sans-serif; font-size: 11px; fill: #888; }
  </style>
  <marker id="arr-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="#888" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>

<!-- L1 -->
<rect x="20" y="20" width="580" height="58" rx="10" class="layer-coral" stroke-width="0.5"/>
<text class="th" x="36" y="44">L1 · Agent system</text>
<text class="ts" x="36" y="64">目标分解 · 会话管理 · 长期记忆 · 多 Agent 协调</text>
<text class="label-r" x="610" y="44">CrewAI</text>
<text class="label-r" x="610" y="58">LangGraph</text>

<!-- L2 -->
<rect x="36" y="94" width="548" height="58" rx="10" class="layer-purple" stroke-width="0.5"/>
<text class="th" x="52" y="118">L2 · Agent loop</text>
<text class="ts" x="52" y="138">Thought → Action → Observation 循环 · 终止条件判断</text>
<text class="label-r" x="594" y="118">AgentExecutor</text>

<!-- L3 -->
<rect x="52" y="168" width="516" height="58" rx="10" class="layer-blue" stroke-width="0.5"/>
<text class="th" x="68" y="192">L3 · Context assembly</text>
<text class="ts" x="68" y="212">system prompt + 历史消息 + 工具定义 + 检索结果 → messages[]</text>
<text class="label-r" x="578" y="192">PromptTemplate</text>

<!-- L4 -->
<rect x="68" y="242" width="484" height="58" rx="10" class="layer-amber" stroke-width="0.5"/>
<text class="th" x="84" y="266">L4 · LLM inference call</text>
<text class="ts" x="84" y="286">POST /v1/messages · model · max_tokens · stream · tools</text>
<text class="label-r" x="562" y="266">Anthropic SDK</text>

<!-- L5 -->
<rect x="84" y="316" width="452" height="58" rx="10" class="layer-teal" stroke-width="0.5"/>
<text class="th" x="100" y="340">L5 · Response parsing</text>
<text class="ts" x="100" y="360">text / tool_use 块解析 · stop_reason 判断 · token 计数</text>
<text class="label-r" x="546" y="340">OutputParser</text>

<!-- L6 -->
<rect x="100" y="390" width="420" height="58" rx="10" class="layer-green" stroke-width="0.5"/>
<text class="th" x="116" y="414">L6 · Tool execution</text>
<text class="ts" x="116" y="434">函数调用 · 外部 API · 代码执行 · 文件读写 · 结果序列化</text>
<text class="label-r" x="530" y="414">@tool / MCP</text>

<!-- L7 -->
<rect x="116" y="464" width="388" height="42" rx="10" class="layer-gray" stroke-width="0.5"/>
<text class="th" x="132" y="480">L7 · Observation write-back</text>
<text class="ts" x="132" y="497">工具结果追加到 messages[] · 触发下一轮 L3→L4</text>

<!-- vertical arrows -->
<line x1="30" y1="78" x2="30" y2="94" class="arr" marker-end="url(#arr-b)"/>
<line x1="46" y1="152" x2="46" y2="168" class="arr" marker-end="url(#arr-b)"/>
<line x1="62" y1="226" x2="62" y2="242" class="arr" marker-end="url(#arr-b)"/>
<line x1="78" y1="300" x2="78" y2="316" class="arr" marker-end="url(#arr-b)"/>
<line x1="94" y1="374" x2="94" y2="390" class="arr" marker-end="url(#arr-b)"/>
<line x1="110" y1="448" x2="110" y2="464" class="arr" marker-end="url(#arr-b)"/>

<!-- feedback loop -->
<path d="M 504 485 L 650 485 L 650 200 L 568 200" fill="none" stroke="#aaa" stroke-width="0.5" stroke-dasharray="5 3" marker-end="url(#arr-b)"/>
<text class="label-r" x="656" y="345">↻ 下一轮</text>
</svg>

**最关键的一条认知**：LLM（L4）是整个栈里唯一真正"思考"的地方，但它是**无状态的**。Agent 的所有记忆、状态、历史，都存在调用者维护的 `messages[]` 里（L3）。框架的工作，就是精心管理这个数组。

---

## 三、一次完整的执行轨迹

以"查询最新量子计算进展并总结"为例，追踪数据在七层间的流动：

```
L1  接收任务，分配给 ResearchAgent
  └─ L2  启动循环（while not done, max_iterations=10）
       └─ L3  组装 messages[]：
              system: "你是研究员，可用工具：web_search"
              user:   "查最新量子计算进展"
            └─ L4  POST /v1/messages  ← 第1次 LLM 调用
                   stop_reason: "tool_use"
                   content: {name:"web_search", input:{query:"量子计算 2026"}}
                 └─ L5  解析出 tool_use block
                      └─ L6  执行 web_search("量子计算 2026")
                           └─ L7  结果追加到 messages[]
                                  ↻ 回到 L3（第2轮）
            └─ L4  POST /v1/messages  ← 第2次 LLM 调用
                   stop_reason: "end_turn"
                   content: {type:"text", text:"根据最新资料…"}
  └─ L2  stop_reason == end_turn，退出循环
└─ L1  收集结果，传递给下一个 Agent
```

一次"Agent 完成任务"，在底层往往是 **3～15 次独立的 HTTP 请求**。每次请求都是从零开始的推理，上下文靠 `messages[]` 传递。

---

## 四、常见 Workflow 模式详解

### 链式（Sequential Chain）

最基础的线性结构，每步输出作为下一步输入：

<svg width="100%" viewBox="0 0 680 100" xmlns="http://www.w3.org/2000/svg">
<defs>
  <style>
    .th { font-family: sans-serif; font-size: 13px; font-weight: 500; fill: #1a1a1a; }
    .ts { font-family: sans-serif; font-size: 11px; fill: #555; }
  </style>
  <marker id="a1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="#888" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>
<rect x="20" y="28" width="100" height="44" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text class="th" x="70" y="47" text-anchor="middle">用户输入</text>

<line x1="120" y1="50" x2="152" y2="50" stroke="#888" stroke-width="1.5" marker-end="url(#a1)"/>

<rect x="152" y="28" width="120" height="44" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
<text class="th" x="212" y="45" text-anchor="middle">Step 1</text>
<text class="ts" x="212" y="62" text-anchor="middle">提取关键词</text>

<line x1="272" y1="50" x2="304" y2="50" stroke="#888" stroke-width="1.5" marker-end="url(#a1)"/>

<rect x="304" y="28" width="120" height="44" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
<text class="th" x="364" y="45" text-anchor="middle">Step 2</text>
<text class="ts" x="364" y="62" text-anchor="middle">检索文档</text>

<line x1="424" y1="50" x2="456" y2="50" stroke="#888" stroke-width="1.5" marker-end="url(#a1)"/>

<rect x="456" y="28" width="120" height="44" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
<text class="th" x="516" y="45" text-anchor="middle">Step 3</text>
<text class="ts" x="516" y="62" text-anchor="middle">生成回答</text>

<line x1="576" y1="50" x2="608" y2="50" stroke="#888" stroke-width="1.5" marker-end="url(#a1)"/>

<rect x="608" y="28" width="56" height="44" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text class="th" x="636" y="50" text-anchor="middle">输出</text>
</svg>

**代表实现**：LangChain LCEL `prompt | llm | parser`。适合步骤确定、边界清晰的任务（RAG 问答、文档摘要）。

---

### ReAct（推理-行动循环）

每步先推理，决定工具，观察结果，再推理下一步：

<svg width="100%" viewBox="0 0 680 180" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="a2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="#888" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>
<rect x="20" y="68" width="90" height="44" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="13" font-weight="500" fill="#1a1a1a" x="65" y="90" text-anchor="middle">问题</text>

<line x1="110" y1="90" x2="142" y2="90" stroke="#888" stroke-width="1.5" marker-end="url(#a2)"/>

<rect x="142" y="58" width="120" height="64" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="13" font-weight="500" fill="#3C3489" x="202" y="82" text-anchor="middle">Thought</text>
<text font-family="sans-serif" font-size="11" fill="#534AB7" x="202" y="100" text-anchor="middle">LLM 推理</text>
<text font-family="sans-serif" font-size="11" fill="#534AB7" x="202" y="114" text-anchor="middle">下一步计划</text>

<line x1="262" y1="90" x2="294" y2="90" stroke="#888" stroke-width="1.5" marker-end="url(#a2)"/>

<rect x="294" y="58" width="120" height="64" rx="8" fill="#FAEEDA" stroke="#854F0B" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="13" font-weight="500" fill="#633806" x="354" y="82" text-anchor="middle">Action</text>
<text font-family="sans-serif" font-size="11" fill="#854F0B" x="354" y="100" text-anchor="middle">调用工具</text>
<text font-family="sans-serif" font-size="11" fill="#854F0B" x="354" y="114" text-anchor="middle">搜索 / 代码</text>

<line x1="414" y1="90" x2="446" y2="90" stroke="#888" stroke-width="1.5" marker-end="url(#a2)"/>

<rect x="446" y="58" width="120" height="64" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="13" font-weight="500" fill="#085041" x="506" y="82" text-anchor="middle">Observation</text>
<text font-family="sans-serif" font-size="11" fill="#0F6E56" x="506" y="100" text-anchor="middle">获取结果</text>
<text font-family="sans-serif" font-size="11" fill="#0F6E56" x="506" y="114" text-anchor="middle">加入上下文</text>

<path d="M566 90 L620 90 L620 28 L202 28 L202 58" fill="none" stroke="#aaa" stroke-width="0.5" stroke-dasharray="5 3" marker-end="url(#a2)"/>
<text font-family="sans-serif" font-size="11" fill="#888" x="410" y="20" text-anchor="middle">↻ 继续推理直到完成</text>

<line x1="566" y1="112" x2="610" y2="150" stroke="#0F6E56" stroke-width="1.5" marker-end="url(#a2)"/>
<rect x="570" y="140" width="100" height="34" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="12" font-weight="500" fill="#085041" x="620" y="157" text-anchor="middle">Final Answer</text>
</svg>

ReAct 是大多数现代 Agent 框架的默认模式。`stop_reason == "tool_use"` 时循环继续，`stop_reason == "end_turn"` 时任务完成。

---

### Plan-and-Execute

先由 Planner 拆解子任务，再逐步执行，最后合并：

<svg width="100%" viewBox="0 0 680 140" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="a3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="#888" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>
<rect x="20" y="48" width="90" height="44" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="12" font-weight="500" fill="#1a1a1a" x="65" y="68" text-anchor="middle">复杂目标</text>

<line x1="110" y1="70" x2="142" y2="70" stroke="#888" stroke-width="1.5" marker-end="url(#a3)"/>

<rect x="142" y="38" width="110" height="64" rx="8" fill="#FAECE7" stroke="#993C1D" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="13" font-weight="500" fill="#712B13" x="197" y="60" text-anchor="middle">Planner</text>
<text font-family="sans-serif" font-size="11" fill="#993C1D" x="197" y="78" text-anchor="middle">分解为</text>
<text font-family="sans-serif" font-size="11" fill="#993C1D" x="197" y="93" text-anchor="middle">子任务列表</text>

<line x1="252" y1="60" x2="284" y2="28" stroke="#D85A30" stroke-width="1.5" marker-end="url(#a3)"/>
<line x1="252" y1="70" x2="284" y2="70" stroke="#D85A30" stroke-width="1.5" marker-end="url(#a3)"/>
<line x1="252" y1="80" x2="284" y2="112" stroke="#D85A30" stroke-width="1.5" marker-end="url(#a3)"/>

<rect x="284" y="10" width="100" height="36" rx="6" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="12" font-weight="500" fill="#3C3489" x="334" y="28" text-anchor="middle">子任务 1</text>

<rect x="284" y="52" width="100" height="36" rx="6" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="12" font-weight="500" fill="#3C3489" x="334" y="70" text-anchor="middle">子任务 2</text>

<rect x="284" y="94" width="100" height="36" rx="6" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="12" font-weight="500" fill="#3C3489" x="334" y="112" text-anchor="middle">子任务 3</text>

<line x1="384" y1="28" x2="440" y2="68" stroke="#7F77DD" stroke-width="1.5" marker-end="url(#a3)"/>
<line x1="384" y1="70" x2="440" y2="70" stroke="#7F77DD" stroke-width="1.5" marker-end="url(#a3)"/>
<line x1="384" y1="112" x2="440" y2="72" stroke="#7F77DD" stroke-width="1.5" marker-end="url(#a3)"/>

<rect x="440" y="48" width="100" height="44" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="12" font-weight="500" fill="#085041" x="490" y="66" text-anchor="middle">合并</text>
<text font-family="sans-serif" font-size="11" fill="#0F6E56" x="490" y="82" text-anchor="middle">Synthesizer</text>

<line x1="540" y1="70" x2="572" y2="70" stroke="#888" stroke-width="1.5" marker-end="url(#a3)"/>
<rect x="572" y="48" width="90" height="44" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="12" font-weight="500" fill="#1a1a1a" x="617" y="70" text-anchor="middle">最终输出</text>
</svg>

适合长篇报告、复杂研究任务。Planner 和 Executor 可以使用不同的模型（例如用 Opus 规划、用 Haiku 执行），在质量和成本间取得平衡。

---

## 五、Workflow 可以相互嵌套和转换

这些模式不是互斥的。有三种方式将它们组合：

**Prompt 注入**：在 ReAct 的 system prompt 里声明"先规划再执行"的格式，LLM 会在 Thought 里自然产生规划行为。实现简单，适合原型阶段。

**工具封装子 Workflow**：把 Plan-and-Execute 封装成一个工具，注册到 ReAct Agent 里。ReAct 通过推理决定什么时候启动这个子流程。

**状态机显式编排**（LangGraph 风格）：用有向图声明节点和条件边，workflow 切换由图结构保证，不依赖 LLM 的自律性。

一个典型的生产级嵌套：

```
外层：Plan-and-Execute（把任务拆成几章）
  └─ 每章内部：ReAct（动态搜索 + 引用）
       └─ 每章写完：Reflection（Critic 审查，不通过则重写）
```

---

## 六、插件扩展 vs 自建框架：控制力差异

这是整个讨论里最实用的部分。我们用控制力热力图来可视化两种方式的差异：

<svg width="100%" viewBox="0 0 680 340" xmlns="http://www.w3.org/2000/svg">
<defs>
  <style>
    .lbl { font-family: sans-serif; font-size: 12px; font-weight: 500; fill: #1a1a1a; }
    .sub { font-family: sans-serif; font-size: 11px; fill: #666; }
    .pct { font-family: sans-serif; font-size: 11px; }
  </style>
</defs>

<!-- headers -->
<text class="sub" x="360" y="18">插件扩展</text>
<text class="sub" x="530" y="18">自建框架</text>
<line x1="340" y1="28" x2="340" y2="330" stroke="#ddd" stroke-width="0.5"/>

<!-- rows -->
<!-- L1 -->
<text class="lbl" x="16" y="62">L1 Agent system</text>
<text class="sub" x="16" y="78">目标·协调·记忆</text>
<rect x="348" y="48" width="160" height="34" rx="4" fill="#f0f0f0"/>
<rect x="348" y="48" width="48" height="34" rx="4" fill="#AFA9EC"/>
<text class="pct" x="430" y="68" text-anchor="middle" fill="#3C3489">30%</text>
<rect x="520" y="48" width="140" height="34" rx="4" fill="#f0f0f0"/>
<rect x="520" y="48" width="133" height="34" rx="4" fill="#D85A30"/>
<text class="pct" x="590" y="68" text-anchor="middle" fill="#4A1B0C">95%</text>

<!-- L2 -->
<text class="lbl" x="16" y="112">L2 Agent loop</text>
<text class="sub" x="16" y="128">推理控制·终止条件</text>
<rect x="348" y="98" width="160" height="34" rx="4" fill="#f0f0f0"/>
<rect x="348" y="98" width="56" height="34" rx="4" fill="#AFA9EC"/>
<text class="pct" x="430" y="118" text-anchor="middle" fill="#3C3489">35%</text>
<rect x="520" y="98" width="140" height="34" rx="4" fill="#f0f0f0"/>
<rect x="520" y="98" width="133" height="34" rx="4" fill="#D85A30"/>
<text class="pct" x="590" y="118" text-anchor="middle" fill="#4A1B0C">95%</text>

<!-- L3 -->
<text class="lbl" x="16" y="162">L3 Context assembly</text>
<text class="sub" x="16" y="178">messages[] 构造</text>
<rect x="348" y="148" width="160" height="34" rx="4" fill="#f0f0f0"/>
<rect x="348" y="148" width="80" height="34" rx="4" fill="#AFA9EC"/>
<text class="pct" x="430" y="168" text-anchor="middle" fill="#3C3489">50%</text>
<rect x="520" y="148" width="140" height="34" rx="4" fill="#f0f0f0"/>
<rect x="520" y="148" width="140" height="34" rx="4" fill="#D85A30"/>
<text class="pct" x="590" y="168" text-anchor="middle" fill="#4A1B0C">100%</text>

<!-- L4 -->
<text class="lbl" x="16" y="212">L4 LLM call</text>
<text class="sub" x="16" y="228">模型推理（双方均无法控制内部）</text>
<rect x="348" y="198" width="160" height="34" rx="4" fill="#f0f0f0"/>
<rect x="348" y="198" width="16" height="34" rx="4" fill="#AFA9EC"/>
<text class="pct" x="430" y="218" text-anchor="middle" fill="#888">10%</text>
<rect x="520" y="198" width="140" height="34" rx="4" fill="#f0f0f0"/>
<rect x="520" y="198" width="21" height="34" rx="4" fill="#D85A30"/>
<text class="pct" x="590" y="218" text-anchor="middle" fill="#888">15%</text>

<!-- L5 -->
<text class="lbl" x="16" y="262">L5 Response parsing</text>
<text class="sub" x="16" y="278">输出解析·结构提取</text>
<rect x="348" y="248" width="160" height="34" rx="4" fill="#f0f0f0"/>
<rect x="348" y="248" width="112" height="34" rx="4" fill="#AFA9EC"/>
<text class="pct" x="430" y="268" text-anchor="middle" fill="#3C3489">70%</text>
<rect x="520" y="248" width="140" height="34" rx="4" fill="#f0f0f0"/>
<rect x="520" y="248" width="140" height="34" rx="4" fill="#D85A30"/>
<text class="pct" x="590" y="268" text-anchor="middle" fill="#4A1B0C">100%</text>

<!-- L6/L7 -->
<text class="lbl" x="16" y="312">L6/L7 Tool execution &amp; write-back</text>
<text class="sub" x="16" y="328">插件的主战场</text>
<rect x="348" y="298" width="160" height="34" rx="4" fill="#f0f0f0"/>
<rect x="348" y="298" width="144" height="34" rx="4" fill="#AFA9EC"/>
<text class="pct" x="430" y="318" text-anchor="middle" fill="#3C3489">90%</text>
<rect x="520" y="298" width="140" height="34" rx="4" fill="#f0f0f0"/>
<rect x="520" y="298" width="140" height="34" rx="4" fill="#D85A30"/>
<text class="pct" x="590" y="318" text-anchor="middle" fill="#4A1B0C">100%</text>

<!-- legend -->
<rect x="348" y="8" width="12" height="10" rx="2" fill="#AFA9EC"/>
<rect x="520" y="8" width="12" height="10" rx="2" fill="#D85A30"/>
</svg>

规律一目了然：**插件扩展的控制力集中在 L6（工具执行），自建框架的增量控制力在 L1-L3（循环上层）**。

---

## 七、Prompt 能控制循环结构吗？

这是最常被问到的问题，也最容易产生误解。

直接说结论：**Prompt 控制的是 LLM 说什么，代码控制的是程序做什么。两件事在正常情况下看起来一样，在边缘情况下完全不同。**

<svg width="100%" viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg">
<defs>
  <marker id="a4" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="#888" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
</defs>

<!-- prompt side -->
<text font-family="sans-serif" font-size="13" font-weight="500" fill="#1a1a1a" x="160" y="20" text-anchor="middle">Prompt 控制</text>
<rect x="20" y="30" width="280" height="36" rx="6" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="12" fill="#3C3489" x="160" y="52" text-anchor="middle">system: "先规划，再逐步执行…"</text>
<line x1="160" y1="66" x2="160" y2="86" stroke="#888" stroke-width="1.5" marker-end="url(#a4)"/>
<rect x="20" y="86" width="280" height="36" rx="6" fill="#FAEEDA" stroke="#854F0B" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="12" fill="#633806" x="160" y="108" text-anchor="middle">LLM Thought: "我将分3步…"（文本）</text>
<line x1="160" y1="122" x2="160" y2="142" stroke="#888" stroke-width="1.5" marker-end="url(#a4)"/>
<rect x="20" y="142" width="280" height="44" rx="6" fill="#FCEBEB" stroke="#A32D2D" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="12" fill="#501313" x="160" y="160" text-anchor="middle">框架只看 stop_reason</text>
<text font-family="sans-serif" font-size="11" fill="#791F1F" x="160" y="178" text-anchor="middle">不执行 LLM 描述的拓扑</text>

<!-- divider -->
<line x1="340" y1="20" x2="340" y2="195" stroke="#ddd" stroke-width="0.5" stroke-dasharray="4 3"/>

<!-- code side -->
<text font-family="sans-serif" font-size="13" font-weight="500" fill="#1a1a1a" x="510" y="20" text-anchor="middle">代码控制</text>
<rect x="360" y="30" width="300" height="36" rx="6" fill="#FAECE7" stroke="#993C1D" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="12" fill="#712B13" x="510" y="52" text-anchor="middle">plan = planner_llm.invoke(task)</text>
<line x1="510" y1="66" x2="510" y2="86" stroke="#D85A30" stroke-width="1.5" marker-end="url(#a4)"/>
<rect x="360" y="86" width="300" height="36" rx="6" fill="#FAECE7" stroke="#993C1D" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="12" fill="#712B13" x="510" y="108" text-anchor="middle">for step in plan.steps: execute(step)</text>
<line x1="510" y1="122" x2="510" y2="142" stroke="#1D9E75" stroke-width="1.5" marker-end="url(#a4)"/>
<rect x="360" y="142" width="300" height="44" rx="6" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
<text font-family="sans-serif" font-size="12" fill="#085041" x="510" y="160" text-anchor="middle">循环由 Python 代码保证</text>
<text font-family="sans-serif" font-size="11" fill="#0F6E56" x="510" y="178" text-anchor="middle">不依赖 LLM 记住计划</text>
</svg>

Prompt 控制在以下场景会系统性失效：

- **步骤间有条件跳转**：LLM 会在 Thought 里写出判断，但框架不执行它
- **需要真正的并行**：LLM 说"并行执行"，但框架的循环是单线程的
- **长任务的上下文压力**：第15步时，前面的计划文本可能已被截断
- **失败需要具体补救逻辑**：框架只会重试整个循环，不执行你写在 prompt 里的 fallback

**判断标准**：你对这个"拓扑"的执行，是否需要确定性保证？

```
需要确定性 → 代码控制 L2（自建或 LangGraph）
不需要确定性 → Prompt 控制完全够用，自建是过度工程
```

---

## 八、何时真的需要自建 Agent

基于以上分析，触发自建的真实条件是：**你的需求在哪些层上与框架的默认行为产生了不可调和的结构性冲突**。

| 冲突层级 | 具体表现 | 插件能解决？ | 结论 |
|---------|---------|------------|------|
| L6 工具执行 | 需要调用内部 API、自定义工具逻辑 | ✅ 能，这正是插件的设计目的 | 用插件扩展 |
| L5 输出解析 | 需要非标结构化输出、流式处理 | ⚠️ 部分能 | 视复杂度决定 |
| L3 上下文组装 | 超长上下文精细压缩、RAG 注入位置影响效果 | ❌ 不能，框架的 memory 是固定的 | 需要自建 L3+ |
| L2 循环结构 | 非 ReAct 拓扑、动态模型切换、自定义失败策略 | ❌ 不能，框架循环不可重塑 | 需要自建 L2+ |
| L1 协调审计 | 合规审计每次调用、自定义 Agent 拓扑 | ❌ 不能，框架协调机制不可替换 | 必须自建整个栈 |

**一个快速自检问题**：你所有的需求，是否都能表达为"我需要一个新工具"或"我需要修改输出格式"？

- 如果是 → L6/L5 问题，插件扩展足够，自建只会引入不必要的维护成本
- 如果否 → 找到具体是哪层产生了冲突，那一层及以上需要自建

---

## 九、总结：一张思维地图

```
用户需求
   │
   ▼
需求能否用 L6（新工具）满足？
   │ 是 → 插件扩展，五分钟搞定
   │ 否 ↓
需求是否涉及 L3（上下文控制）的精度？
   │ 否 → 调整 prompt，验证效果
   │ 是 ↓
需求是否涉及 L2（循环结构）的确定性？
   │ 否 → Fork 框架，局部定制 L3
   │ 是 ↓
需求是否涉及 L1（协调/审计/合规）？
   │ 否 → 自建 L2+L3，复用框架的 L1
   │ 是 → 自建整个栈
```

最后一个值得反复记住的洞察：**LLM 本身是无状态的，Agent 的"智能"由框架对 `messages[]` 的管理决定**。理解这一点，就理解了为什么不同的框架在面对同样的 LLM 时，会产生如此不同的行为——它们的差异不在模型，而在 L1 到 L3 这三层的设计哲学。

---

*本文整理自一次关于 AI Agent 架构的深度讨论，覆盖了从 Workflow 模式到执行层级、从插件扩展到自建框架的完整认知链路。*
