"""RL 强化学习模块 — Search-R1 + WebWalker 垂直搜索。

Ported from XIAOMI_SU7_RAG/src/rl/.

工具协议（5 个标签）:
  <search_local>关键词</search_local>      — 本地知识库检索
  <search_web>关键词</search_web>          — 网络搜索
  <read_page>URL</read_page>               — 深度阅读网页内容
  <information>检索结果</information>       — 系统注入的检索信息
  <answer>最终答案</answer>                — 模型生成的答案

三级检索路由：
  local → web → read_page，模型自主决定何时检索、是否深度阅读

6 维奖励函数（总分 1.0）：
  1. 格式完整性  0.05 — 标签齐全、正确闭合
  2. 答案质量    0.40 — 准确性、信息量、groundedness
  3. 工具合理性  0.15 — 关键词精准、调用顺序合理
  4. 来源标注    0.10 — 网络信息注明来源
  5. 领域合规    0.15 — 非SU7问题正确拒答
  6. 探索深度    0.15 — 本地充足即停 / 有效利用web/read_page
"""
