---
title: 内嵌 SVG 代码测试
date: 2026-03-19 10:00:00 +0800
categories: [测试]
tags: [SVG, 图片]
---

测试在 Markdown 中直接内嵌 SVG 源码是否可以渲染。

## 内嵌 SVG 代码

<svg width="100%" viewBox="0 0 680 390" xmlns="http://www.w3.org/2000/svg">
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
  <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
<style>
  .t  { font-family: sans-serif; font-size: 14px; font-weight: 400; fill: #2C2C2A; }
  .ts { font-family: sans-serif; font-size: 12px; font-weight: 400; fill: #5F5E5A; }
  .th { font-family: sans-serif; font-size: 14px; font-weight: 500; fill: #2C2C2A; }
  .arr { stroke: #888780; stroke-width: 1.5; fill: none; }

  .c-purple rect, .c-purple ellipse, .c-purple circle { fill: #EEEDFE; stroke: #534AB7; }
  .c-purple .th { fill: #3C3489; }
  .c-purple .ts { fill: #534AB7; }

  .c-teal rect, .c-teal ellipse, .c-teal circle { fill: #E1F5EE; stroke: #0F6E56; }
  .c-teal .th { fill: #085041; }
  .c-teal .ts { fill: #0F6E56; }

  .c-amber rect, .c-amber ellipse, .c-amber circle { fill: #FAEEDA; stroke: #854F0B; }
  .c-amber .th { fill: #633806; }
  .c-amber .ts { fill: #854F0B; }

  .c-coral rect, .c-coral ellipse, .c-coral circle { fill: #FAECE7; stroke: #993C1D; }
  .c-coral .th { fill: #712B13; }
  .c-coral .ts { fill: #993C1D; }

  .c-gray rect, .c-gray ellipse, .c-gray circle { fill: #F1EFE8; stroke: #5F5E5A; }
  .c-gray .th { fill: #444441; }
  .c-gray .ts { fill: #5F5E5A; }

  @media (prefers-color-scheme: dark) {
    .t  { fill: #C2C0B6; }
    .ts { fill: #888780; }
    .th { fill: #D3D1C7; }
    .arr { stroke: #888780; }

    .c-purple rect, .c-purple ellipse, .c-purple circle { fill: #3C3489; stroke: #AFA9EC; }
    .c-purple .th { fill: #CECBF6; }
    .c-purple .ts { fill: #AFA9EC; }

    .c-teal rect, .c-teal ellipse, .c-teal circle { fill: #085041; stroke: #5DCAA5; }
    .c-teal .th { fill: #9FE1CB; }
    .c-teal .ts { fill: #5DCAA5; }

    .c-amber rect, .c-amber ellipse, .c-amber circle { fill: #633806; stroke: #EF9F27; }
    .c-amber .th { fill: #FAC775; }
    .c-amber .ts { fill: #EF9F27; }

    .c-coral rect, .c-coral ellipse, .c-coral circle { fill: #712B13; stroke: #F0997B; }
    .c-coral .th { fill: #F5C4B3; }
    .c-coral .ts { fill: #F0997B; }

    .c-gray rect, .c-gray ellipse, .c-gray circle { fill: #444441; stroke: #B4B2A9; }
    .c-gray .th { fill: #D3D1C7; }
    .c-gray .ts { fill: #B4B2A9; }
  }
</style>
</defs>

<!-- Rollout box -->
<g class="c-purple">
  <rect x="40" y="40" width="180" height="80" rx="8" stroke-width="0.5"/>
  <text class="th" x="130" y="68" text-anchor="middle" dominant-baseline="central">1次 Rollout</text>
  <text class="ts" x="130" y="88" text-anchor="middle" dominant-baseline="central">G个response</text>
  <text class="ts" x="130" y="104" text-anchor="middle" dominant-baseline="central">存好 old_log_probs</text>
</g>

<!-- Arrow to update loop -->
<line x1="220" y1="80" x2="258" y2="80" class="arr" marker-end="url(#arrow)"/>

<!-- K updates dashed container -->
<rect x="260" y="30" width="370" height="160" rx="10" fill="none" stroke="#888780" stroke-width="0.5" stroke-dasharray="5 3"/>
<text class="ts" x="280" y="48" dominant-baseline="central">重复使用同一批数据，共 K 次</text>

<g class="c-teal">
  <rect x="278" y="56" width="100" height="56" rx="8" stroke-width="0.5"/>
  <text class="th" x="328" y="78" text-anchor="middle" dominant-baseline="central">update 1</text>
  <text class="ts" x="328" y="96" text-anchor="middle" dominant-baseline="central">θ → θ'</text>
</g>
<g class="c-teal">
  <rect x="400" y="56" width="100" height="56" rx="8" stroke-width="0.5"/>
  <text class="th" x="450" y="78" text-anchor="middle" dominant-baseline="central">update 2</text>
  <text class="ts" x="450" y="96" text-anchor="middle" dominant-baseline="central">θ' → θ''</text>
</g>
<g class="c-teal">
  <rect x="522" y="56" width="90" height="56" rx="8" stroke-width="0.5"/>
  <text class="th" x="567" y="78" text-anchor="middle" dominant-baseline="central">... K</text>
  <text class="ts" x="567" y="96" text-anchor="middle" dominant-baseline="central">θ^(K)</text>
</g>

<line x1="378" y1="84" x2="398" y2="84" class="arr" marker-end="url(#arrow)"/>
<line x1="500" y1="84" x2="520" y2="84" class="arr" marker-end="url(#arrow)"/>

<text class="ts" x="445" y="145" text-anchor="middle">old_log_probs 全程不变，ratio 随 θ 漂移而增大</text>
<text class="ts" x="445" y="162" text-anchor="middle">clip(ρ, 1−ε, 1+ε) 防止单步更新过猛</text>

<!-- Sync arrow -->
<line x1="445" y1="192" x2="445" y2="228" class="arr" marker-end="url(#arrow)"/>
<text class="ts" x="470" y="216" dominant-baseline="central">K步后同步</text>

<!-- Sync box -->
<g class="c-gray">
  <rect x="310" y="230" width="270" height="44" rx="8" stroke-width="0.5"/>
  <text class="th" x="445" y="252" text-anchor="middle" dominant-baseline="central">θ_old ← θ^(K)，重新 Rollout</text>
</g>

<!-- Divider -->
<line x1="40" y1="300" x2="640" y2="300" stroke="#888780" stroke-width="0.5" stroke-dasharray="4 3"/>

<!-- K values table -->
<text class="th" x="40" y="326" dominant-baseline="central">典型 K 值（不同实现）</text>

<g class="c-amber">
  <rect x="40" y="340" width="130" height="36" rx="6" stroke-width="0.5"/>
  <text class="th" x="105" y="352" text-anchor="middle" dominant-baseline="central">DeepSeek-R1</text>
  <text class="ts" x="105" y="368" text-anchor="middle" dominant-baseline="central">K = 1（每次重采）</text>
</g>
<g class="c-teal">
  <rect x="186" y="340" width="130" height="36" rx="6" stroke-width="0.5"/>
  <text class="th" x="251" y="352" text-anchor="middle" dominant-baseline="central">PPO 经典设置</text>
  <text class="ts" x="251" y="368" text-anchor="middle" dominant-baseline="central">K = 4 ~ 10</text>
</g>
<g class="c-purple">
  <rect x="332" y="340" width="130" height="36" rx="6" stroke-width="0.5"/>
  <text class="th" x="397" y="352" text-anchor="middle" dominant-baseline="central">GRPO 原论文</text>
  <text class="ts" x="397" y="368" text-anchor="middle" dominant-baseline="central">K = 1 ~ 4</text>
</g>
<g class="c-coral">
  <rect x="478" y="340" width="160" height="36" rx="6" stroke-width="0.5"/>
  <text class="th" x="558" y="352" text-anchor="middle" dominant-baseline="central">K 增大的代价</text>
  <text class="ts" x="558" y="368" text-anchor="middle" dominant-baseline="central">ratio 偏差积累，不稳定</text>
</g>
</svg>
