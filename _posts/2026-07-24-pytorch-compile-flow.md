---
title: PyTorch Compile 流程讲解
date: 2026-07-24 00:00:00 +0800
categories: [AI 框架, 编译器]
tags: [PyTorch, torch.compile, Inductor, 编译器]
description: 面向自研硬件 runtime 开发者的 torch.compile 编译流程源码级讲解
---

> 面向自研硬件 runtime 开发者的 `torch.compile` 编译流程源码级讲解：Dynamo 与 AOTAutograd 只做入口铺垫，重点落在 Inductor——各阶段"图"的表达形态、如何解析节点信息、融合算子从定义到释放的完整生命周期，以及如何为自研芯片注册 backend。

<iframe id="dc-doc"
        src="{{ '/assets/pytorch-compile/pytorch-compile.dc.html' | relative_url }}"
        title="PyTorch Compile 流程讲解"
        loading="lazy"
        scrolling="no"
        style="width:100%;border:0;min-height:600px;"></iframe>

<script>
(function () {
  var f = document.getElementById('dc-doc');
  if (!f) return;
  var last = 0;

  function measure() {
    var d;
    try { d = f.contentDocument; } catch (e) { return; } // 跨域保护，本站同源不触发
    if (!d || !d.body) return;
    // 文档的 support.js 会注入 html,body{height:100%} 与 #dc-root{height:100%}，
    // 即 body 盒子被钉死等于 iframe 高度，真实内容只以 scrollHeight（溢出）体现。
    // 把 iframe 高度设为该 scrollHeight 后，body 100% 也随之等于内容高度，
    // scrollHeight 稳定，不会形成撑高的正反馈。
    var h = Math.max(d.body.scrollHeight, d.documentElement.scrollHeight);
    if (h > 0 && Math.abs(h - last) > 2) {
      last = h;
      f.style.height = h + 'px';
    }
  }

  // 关键：文档是 React 客户端渲染（从 CDN 加载 React/Babel 再编译），内容注入时机
  // 不确定；且 body 盒子被钉在 iframe 高度上，尺寸不变，ResizeObserver 观察不到。
  // 因此用 MutationObserver 监听 body 子树——React 注入 DOM 的那一刻必然触发，
  // 无论多晚。measure 读 scrollHeight 得到真实内容高度。
  function schedule() { requestAnimationFrame(measure); }
  function attach() {
    var d;
    try { d = f.contentDocument; } catch (e) { return; }
    if (!d || !d.body) return;
    measure();
    if (window.MutationObserver && !f.__mo) {
      var mo = new MutationObserver(schedule);
      mo.observe(d.body, { childList: true, subtree: true, attributes: true, characterData: true });
      f.__mo = mo;
    }
  }

  f.addEventListener('load', attach);
  attach();                       // body 通常在 React 渲染前就已可读
  // 兜底：若 attach 时 body 尚未就绪，短轮询直到挂上 MutationObserver（挂上即停）
  var guard = setInterval(function () {
    if (f.__mo) { clearInterval(guard); return; }
    attach();
  }, 200);
  setTimeout(function () { clearInterval(guard); }, 30000);
  // 字体异步回流兜底：字体到位后行高变化，MutationObserver 不一定覆盖，补几次采样
  [800, 2500, 6000, 12000].forEach(function (ms) { setTimeout(measure, ms); });
})();
</script>
