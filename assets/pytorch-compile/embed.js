// 同源 iframe 自适应高度：使被嵌入的 pytorch-compile 文档随内容撑满，无内部滚动条。
//
// 为什么必须是外部脚本：Chirpy 的 post.min.js 会对文章 .content 做 innerHTML 级处理
// （TOC、图片包裹、代码块等），通过 innerHTML 重新插入的内联 <script> 不会被浏览器执行。
// 外部 <script src> 即使被重插也会执行，因此把逻辑放这里而非文章正文的内联 <script> 中。
(function () {
  var f = document.getElementById('dc-doc');
  if (!f) return;
  var last = 0;

  function measure() {
    var d;
    try { d = f.contentDocument; } catch (e) { return; } // 跨域保护，本站同源不触发
    if (!d || !d.body) return;
    // 文档的 support.js 注入 html,body{height:100%} 与 #dc-root{height:100%}，
    // body 盒子被钉死等于 iframe 高度，真实内容只以 scrollHeight（溢出）体现。
    // 把 iframe 高度设为该 scrollHeight 后，body 100% 随之等于内容高度，scrollHeight
    // 稳定，不会形成撑高的正反馈。
    var h = Math.max(d.body.scrollHeight, d.documentElement.scrollHeight);
    if (h > 0 && Math.abs(h - last) > 2) {
      last = h;
      f.style.height = h + 'px';
    }
  }

  // 文档是 React 客户端渲染（从 CDN 加载 React/Babel 再编译），内容注入时机不确定；
  // 且 body 盒子被钉在 iframe 高度上、尺寸不变，ResizeObserver 观察不到。改用
  // MutationObserver 监听 body 子树——React 注入 DOM 的那一刻必然触发，无论多晚。
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
  // 字体异步回流兜底：字体到位后行高变化，补几次采样
  [800, 2500, 6000, 12000].forEach(function (ms) { setTimeout(measure, ms); });
})();
