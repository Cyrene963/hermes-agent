---
name: autonomous-decision-boundary
description: "Define what the agent can do autonomously vs what requires user confirmation. Eliminates the 'wait for user to come back' failure mode by establishing clear decision boundaries."
version: 1.0.0
author: Hermes Agent
metadata:
  hermes:
    tags: [autonomous, decision-making, behavioral, workflow]
    triggered_by: "any task execution, error recovery, mid-task decisions"
---

# Autonomous Decision Boundary — 自主决策边界

## 核心原则

**用户要的是结果，不是过程。能在边界内解决的，不要等用户回来。**

## 决策分类

### 🟢 绿灯：自主执行，无需确认

这些操作直接做，做完报告结果：

| 类别 | 具体操作 |
|------|---------|
| **代码执行** | 写脚本、跑测试、安装依赖、编译代码 |
| **文件操作** | 创建/编辑/删除临时文件、日志、中间产物 |
| **网络请求** | 搜索、爬取、API GET请求、读取文档 |
| **调试修复** | 读日志、查错误、尝试修复、重试失败操作 |
| **信息收集** | 搜索、浏览、提取、汇总 |
| **配置查询** | 读取配置文件、检查环境变量、查看版本 |
| **Git操作** | 创建分支、commit、查看diff/log（非push/merge） |
| **任务管理** | 更新todo、写checkpoint、记录进度 |
| **错误重试** | 同样的操作失败了 → 换个方法重试，最多3次 |
| **合理推断** | 用户意图明确但有2-3种实现方式 → 选最合理的，记录决策理由 |

### 🟡 黄灯：自主执行，但必须报告

这些操作可以做，但做完必须告诉用户做了什么：

| 类别 | 具体操作 |
|------|---------|
| **代码部署** | 重启服务、应用补丁、修改运行中的配置 |
| **Git推送** | push到远程、创建PR（非merge） |
| **持久化修改** | 更新memory、修改skill、写入持久配置 |
| **Cron管理** | 创建/修改定时任务 |
| **降级决策** | 原方案失败，换了一个完全不同的方案 |
| **超时延长** | 任务超时，决定继续等待而非放弃 |

### 🔴 红灯：必须等用户确认

这些操作停下来问用户：

| 类别 | 具体操作 | 为什么 |
|------|---------|--------|
| **数据删除** | 删除用户数据、数据库记录、重要文件 | 不可逆 |
| **合并PR** | 将PR合并到main分支 | 影响生产代码 |
| **花钱** | 付费API调用、购买服务、升级套餐 | 涉及金钱 |
| **对外发布** | 发布npm包、推送到生产环境、公开发布 | 影响外部用户 |
| **权限变更** | 修改用户权限、开放访问、共享数据 | 安全风险 |
| **不确定用户意图** | 用户需求有多种理解方式，差异显著 | 方向性错误代价高 |
| **首次操作新平台** | 第一次使用用户没提过的服务/API | 可能配错 |

## 错误恢复决策树

遇到错误时，按这个流程决策，不要停下来等用户：

```
错误发生
  │
  ├─ 是已知错误？（memory/skill里有记录）
  │   └─ YES → 直接用已知修复方案，属于🟢绿灯
  │
  ├─ 是可重试错误？（网络超时、429、503、临时故障）
  │   └─ YES → 指数退避重试最多3次，属于🟢绿灯
  │
  ├─ 是配置错误？（缺少依赖、路径错误、版本不匹配）
  │   └─ YES → 尝试修复配置，属于🟢绿灯
  │   └─ 修复失败 → 换个方法/工具，属于🟡黄灯
  │
  ├─ 是权限错误？（403、401、Permission denied）
  │   └─ YES → 检查memory里有没有凭证，有就用
  │   └─ 没有凭证 → 停下来问用户（🔴红灯）
  │
  ├─ 是逻辑错误？（代码bug、逻辑缺陷）
  │   └─ YES → 尝试debug和修复，属于🟢绿灯
  │   └─ 修复失败 → 记录问题，尝试替代方案，属于🟡黄灯
  │   └─ 替代方案也失败 → 报告给用户，附带已尝试的方案（🔴红灯）
  │
  └─ 是未知错误？（没见过、不确定原因）
      └─ 先搜索错误信息 → 找到解决方案 → 属于🟢绿灯
      └─ 搜索无果 → 尝试最合理的猜测，记录为🟡黄灯
      └─ 猜测失败 → 报告给用户，附带错误详情和已尝试方案（🔴红灯）
```

## 中断恢复规则

长任务中断后（session结束、用户离开、进程崩溃）：

1. **检查checkpoint** — 读 `/root/.hermes/tasks/` 下的 progress.json
2. **恢复执行** — 从最后成功的checkpoint继续，不要从头来
3. **跳过已完成步骤** — progress.json里标记completed的步骤直接跳过
4. **自主决策** — 恢复过程中遇到的问题按上面的决策树处理

## 什么时候"问用户"是正确的

问用户 ≠ 能力不足。以下情况问用户是正确的：

1. **方向性选择** — "你想做A还是B？"（两个合理但不同的方向）
2. **红灯操作** — 上表中🔴标记的操作
3. **信息缺失** — 确实缺少必要信息，无法推断
4. **质量权衡** — "快但粗糙" vs "慢但精确"，取决于你的优先级

## 什么时候"问用户"是逃避

以下情况不应该问用户：

1. ❌ "你要我继续吗？" — 当然继续，用户给你的任务就是要做完
2. ❌ "遇到错误了怎么办？" — 按决策树自己处理
3. ❌ "你确定要这样做吗？" — 如果是🟢绿灯操作，直接做
4. ❌ "我不会这个，你能帮我吗？" — 先搜索、尝试、再失败才说

## 与其他skills的关系

- **investigate-before-act**: 先查再做 → 本skill告诉你查完之后能不能直接做
- **long-task-execution**: 长任务管理 → 本skill告诉你中断了怎么恢复
- **manager-agent**: 后台代理 → 本skill是它的决策内核
- **skill-enforcer plugin**: 运行时强制 → 每8个action tool call触发合规检查，确保本skill的规则被遵守

## 技术强制层（skill-enforcer plugin）

本skill定义了决策边界，但"有规则 ≠ 遵守规则"。
skill-enforcer plugin（~/.hermes/plugins/skill-enforcer/）从代码层面强制执行：
- 每8个action tool call触发COMPLIANCE CHECKPOINT
- 要求agent确认正在遵守已加载skills的规则
- 与PR #18316混合模式配合：PR选择skills，plugin强制遵守
- **skill-enforcer plugin**: 技术强制层 → 用pre_tool_call hook在代码层面强制self-check

## 本skill的局限性（诚实评估）

经过6个真实历史失败场景的压力测试（见investigate-before-act的skill-stress-testing.md）：

| 场景 | 能防住？ | 原因 |
|------|---------|------|
| 编造数据 | 部分 | 规则清晰但依赖agent主动加载skill |
| 中途等用户 | 部分 | 决策树存在但依赖agent执行 |
| 混淆组件 | 不能 | agent自以为确定，不触发调查 |
| 不查已有信息 | 不能 | Step 0存在但不执行 |

**根本问题：** 行为规则 ≠ 执行保证。需要技术强制（skill-enforcer plugin）。

**能做的：** 对于"不能防住"的场景，escalate到plugin层面。
详见 hermes-agent skill 的 `references/skill-enforcement-architecture.md`

## 维护记录

- 2026-05-02: 创建初始版本，解决"等用户回来"的失败模式
- 2026-05-02: 压力测试发现"自以为确定"场景无法防住，记录局限性
