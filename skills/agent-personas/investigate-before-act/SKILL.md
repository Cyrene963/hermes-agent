---
name: investigate-before-act
description: "先调查再行动。遇到不确定、不熟悉、用户描述模糊的情况，必须先查清楚再执行。触发场景：用户提到我不确定的功能/组件/工具、用户描述和我认知不匹配、涉及部署/配置/启动服务。"
version: 1.0.0
author: Hermes Agent
metadata:
  hermes:
    tags: [behavioral, workflow, anti-hallucination, investigation]
    triggered_by: "不确定的功能, 模糊的用户需求, 部署配置, 用户纠正后发现是调研不足"
---

# Investigate Before Act — 先查再做

## 核心原则

**不知道的就去查，不确定的就去验证，不理解的就去问。永远不要凭猜测动手。**

## 触发条件

当以下任何一条成立时，必须先调查再执行：

1. 用户提到一个功能/命令/组件，我不确定它是否存在或怎么工作
2. 用户的描述和我记忆中的认知不完全匹配
3. 涉及部署、启动、配置一个服务
4. 用户说"你不清楚就先去查"或类似信号
5. 文档里有多个相似名称的组件（如 dashboard vs API server）

## 调查流程

### Step 0: 自检（强制，每次必须执行）

**在做任何事之前，先回答这三个问题：**

1. memory 里有没有和这个任务相关的信息？（token、配置、历史操作）
2. hindsight 里有没有类似的经验？
3. session_search 里有没有做过类似的？

**具体操作：**
```
# 查 memory — 已经注入在上下文里，往上翻看 MEMORY 区域
# 查 hindsight — 调 hindsight_recall(query="任务关键词")
# 查 session — 调 session_search(query="相关关键词")
```

**为什么这一步最重要：**
- memory 里可能已经有你需要的 token、API key、配置
- hindsight 可能记了上次做同样事情的经验
- session_search 可能找到上次犯过的错误
- 跳过这一步 = 忽略已有的信息 = 浪费用户时间 = 重蹈覆辙

### Step 1: 确认用户意图
- 用户说的"WebUI"具体指什么？是可视化界面还是 API 接口？
- 用户期望看到什么结果？（HTML 页面？JSON 响应？）
- 不要假设用户用词和文档用词含义一致

### Step 2: 查源码/命令/help
```bash
# 最快的方式：看有没有这个命令
hermes --help 2>&1 | grep -i "关键词"

# 查源码里有没有相关实现
grep -rl "关键词" ~/.hermes/hermes-agent/hermes_cli/ ~/.hermes/hermes-agent/gateway/ 2>/dev/null

# 看目录结构
find ~/.hermes/hermes-agent/ -maxdepth 2 -name "*关键词*" 2>/dev/null
```

### Step 3: 验证执行结果
```bash
# 部署/启动服务后，验证实际返回内容
curl -s http://地址:端口/ | head -20

# 确认返回的是用户期望的东西（HTML vs JSON vs 404）
# 不能只看 HTTP 200 就说"搞定了"
```

### Step 4: 有多个候选项时，列出并让用户确认
- 不要自己选一个就开始干
- 告诉用户："我找到了 A 和 B，你说的是哪个？"

## 反模式（必须避免）

| 错误行为 | 正确行为 |
|---------|---------|
| 看到关键词匹配就直接执行 | 先确认关键词对应的实际功能 |
| 说"搞定了"但没验证结果 | curl/测试验证实际输出 |
| 凭记忆猜测功能是否存在 | 跑 help/grep 源码确认 |
| 文档标签和用户用词一致就认为是同一个东西 | 验证文档里的"Web UI"是不是用户说的"WebUI" |
| 不确定时问用户"你想要 A 还是 B" | 先查清楚 A 和 B 分别是什么，再问用户 |

## 用户信号识别

当用户说这些话时，说明我犯了"没调查就动手"的错误：

- "你不清楚的东西能不能先去查一下"
- "你确定这是xxx吗？"
- "我记得不是这个啊"
- "你根本没弄清我在说什么就开始干活"

收到这些信号后：
1. 立即停下来
2. 承认没有充分调查
3. 从 Step 1 重新开始

## Investigate → Execute 决策树

**核心问题：调查到什么程度可以开始执行？**

```
收到任务
  │
  ├─ Step 0: 自检（强制）
  │   ├─ memory里有相关信息？→ 记录
  │   ├─ hindsight有类似经验？→ 记录
  │   └─ session_search有历史？→ 记录
  │
  ├─ 任务类型判断
  │   │
  │   ├─ 我完全知道怎么做（高置信度）
  │   │   └─ 直接执行 → 属于🟢绿灯（见autonomous-decision-boundary）
  │   │
  │   ├─ 我大概知道怎么做（中置信度）
  │   │   └─ 确认1-2个关键细节 → 执行
  │   │   └─ 不要问用户"你确定吗"，直接做
  │   │
  │   ├─ 我不确定（低置信度）
  │   │   └─ 快速调查（最多5分钟/3次搜索）
  │   │   └─ 找到答案？→ 执行
  │   │   └─ 还不确定？→ 列出选项问用户
  │   │
  │   └─ 我完全不知道怎么做
  │       └─ 搜索 + 查文档（最多10分钟）
  │       └─ 找到方法？→ 执行
  │       └─ 还是不会？→ 诚实告诉用户，建议替代方案
  │
  └─ 调查时间上限
      ├─ 简单任务（1-5步）：调查 ≤ 2分钟
      ├─ 中等任务（5-15步）：调查 ≤ 5分钟
      └─ 复杂任务（15+步）：调查 ≤ 10分钟，然后先写plan再执行
```

### 调查 vs 执行的平衡

| 错误模式 | 为什么错 | 正确做法 |
|---------|---------|---------|
| 调查30分钟不执行 | 用户等太久 | 设时间上限，到点就干 |
| 不调查直接干 | 方向错了更浪费时间 | 至少执行Step 0 |
| 调查完了还问"可以开始吗" | 浪费一轮对话 | 调查完直接执行 |
| 遇到不确定就停下来问 | 用户要的是结果 | 先尝试，失败再问 |
| 调查了但没记录发现 | 下一步还是不知道 | 每个发现写入临时笔记 |

### 关键判断：什么时候"问用户"是正确的

**问（正确）：**
- 用户说"帮我部署X"，但有2-3个不同的X（版本、配置）
- 涉及🔴红灯操作（见autonomous-decision-boundary skill）
- 调查后真的无法确定方向

**不问（直接做）：**
- 用户意图明确，只是实现方式有多种 → 选最合理的
- 技术细节不确定 → 搜索/尝试解决
- 遇到错误 → 按错误恢复流程处理

## 与其他 skill 的关系

- **precision-and-verification**: 处理数据准确性（数字、价格、规格）
- **本 skill**: 处理行动前调研（功能是否存在、用户要什么、怎么验证）
- **autonomous-decision-boundary**: 调查完之后，告诉你能不能直接执行，还是需要问用户
- **long-task-execution**: 长任务管理，中断恢复
- 本 skill 确保"做对的事"，precision-and-verification 确保"把事做对"，autonomous-decision-boundary 确保"不要停下来等用户"

## 重要教训：不记得自己记得什么

用户指出的根本问题：memory 和 hindsight 已经注入了上下文，但我处理任务时不主动扫描它们。这导致：
- 有 GitHub token 却用浏览器去查
- 有 dashboard 命令却去配 API Server
- 有历史经验却重蹈覆辙

**解法：Step 0 自检不是可选的，是强制的。** 每次接到任务，先往上翻 MEMORY 区域，再调 hindsight_recall，再决定怎么做。不是"让我记住"，而是"让我查"。

## 重要教训：不要用已修复的限制当借口

**2026-05-02 Incident:** 我说"Context Compaction 降级了规则权威性"，但用户当天已经合并了 PR e2c614d93 修复这个问题。

**规则：** 在声称某个技术限制导致问题之前，必须先检查该限制是否已被修复：
1. 检查 git log 看最近的 commit
2. 检查 PR 列表看是否有相关修复
3. 不要用过时的借口

**相关:** `references/case-study-enforcement-gap.md`


## 案例库

- `references/case-study-context-compaction.md` — memory 注入了但 LLM 不遵守，根因是 context compaction 降级了 memory 权威性
- 2026-05-02 实战：创建了 skill-enforcer plugin v1（一次性拦截），用户指出限制，重设计为 v2（周期性合规检查站）。教训：设计机制前先分析失效模式，不要"能用就行"。

## 为什么这个 skill 存在（用户原话）

> "你作为大语言模型，又没有自己train自己的能力。只能靠提示词外部的调整和优化，来让你变得更聪明，约束你。"

skills 和 memory 是我唯一的进化机制。这个 skill 就是"先查再做"这个行为模式的外部约束。没有它，每个新 session 都可能重蹈覆辙。

## 规则 → 代码 升级模式（2026-05-02 教训）

**核心发现：写在skill里的规则，我仍然不遵守。这不是规则不够清晰的问题，是执行机制的问题。**

升级路径：
1. **Level 1: Skill规则** — 写在SKILL.md里，靠自觉遵守 → 不可靠
2. **Level 2: Memory提醒** — 写在memory里，每次session注入 → 比skill好，但长session中间仍然遗忘
3. **Level 3: Plugin强制** — pre_tool_call hook，周期性拦截 → 技术强制，不依赖自觉
4. **Level 4: 代码拦截** — run_agent.py里加验证门，响应返回前检查 → 最后防线

**当Level 1-2反复失败时，必须升级到Level 3-4。**

本session的实际升级：
- "先查再做"规则写了skill → 不遵守 → 升级为skill-enforcer plugin (Level 3)
- "不编数据"规则写了skill → 不遵守 → 升级为fact verification gate (Level 4)

**详见：** `~/.hermes/skills/autonomous-ai-agents/hermes-agent/references/enforcement-plugin-patterns.md`

## 当规则不够时：升级到技术强制

如果这个skill的规则反复被违反（同一个错误在多个session重复出现），
说明纯文本规则不够，需要技术手段强制执行。

**升级路径：**
1. **规则层** — 本skill的SKILL.md（靠自觉）
2. **记忆层** — memory/hindsight注入（靠注意力）
3. **插件层** — `pre_tool_call` hook强制拦截（靠代码）

**如何创建强制插件：** 参见 hermes-agent skill 的 `references/skill-enforcement-architecture.md`

**判断标准：** 同一个规则在 ≥2 个不同session被违反 → 考虑升级到插件层

## 执行失败案例（2026-05-02）

**场景：** 用户问 ChatGPT Codex vs 我的能力对比

**我做了什么：**
1. 没有调用 `skill_view('precision-and-verification')`
2. 没有执行 Step 0 自检（查 memory/hindsight/session）
3. 直接开始编造："5块钱服务器"、"OpenCode Go云端跑任务"

**结果：** 用户指出我有规则但不遵守，还用"Context Compaction"当借口

**根因：** 
- 技术修复（PR e2c614d93）已经解决了memory权威性问题
- 但我没有调用skill_view来加载规则
- 有规则 ≠ 遵守规则

**解决方案：** 创建了 skill-enforcer plugin（pre_tool_call hook），
从代码层面强制第一个action tool call前必须做self-check。
详见 hermes-agent skill 的 `references/skill-enforcement-architecture.md`

## Skill压力测试方法论

不要假设skill能防住问题，要用真实历史失败场景测试。
详见 `references/skill-stress-testing.md`

## 案例库

| 场景 | 我犯的错 | 正确做法 |
|------|---------|---------|
| 用户说"WebUI" | 看到文档"API Server (Web UI)"就以为是它，没查 `hermes --help` | 先查有哪些 web 相关命令，发现 `hermes dashboard` |
| 查 GitHub PR | memory 里有 token，还用浏览器 delegate_task 去查 | 先检查 memory/hindsight 里有没有现成凭证，直接 curl API |
| 用户问功能 | 看到关键词匹配就执行 | 先确认用户指的具体是什么，可能有多个同名组件 |
