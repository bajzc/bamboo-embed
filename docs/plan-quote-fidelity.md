# Plan：提升本地小模型的引用保真度与释义质量

> 目标：让 `profile: local`（Qwen3-8B GGUF）在 `guji ask` 上的表现逼近 `qwen-max`，
> **不做任何训练**。四步，每步都有独立的 A/B 观测量，任一步不达标即停下重新评估。
>
> 背景结论（已论证，无需重新推导）：
> - 小模型在本项目上的退化主要不在"文言文知识"（知识由 RAG 提供），而在
>   **长串繁体逐字复制** > **格式/工具跟随** > **释义质量**。
> - `_validate` 的逐字校验是硬约束，模型一旦漂移就触发重试，重试再失败就拒答，
>   所以"复制能力"直接吃掉了引用准确率。

---

## Step 0 — 扩测量（必做前置，无功能改动）

没有分项指标就无法判断后面三步是否有效。**本步不改任何生成逻辑。**

### 0.1 在 `AnswerResult` 上补齐可观测字段

`src/guji/generate.py`：

- `prompt_tokens_max: int` — 每轮 LLM 调用后从 OpenAI response 的 `usage.prompt_tokens`
  取最大值（`_converse` 里累积）。**这是 Step 1 定 ctx-size 的唯一依据。**
- `quote_stats: dict` — `{"total": n, "verbatim_ok": n}`，按**第一次尝试**统计
  （重试后的数字没有诊断价值）。
- 现有的 `attempt_violations` 已经记录了 `(bad_citations, bad_quotes)`，无需新增。

### 0.2 扩 `guji eval` 的指标

`src/guji/eval.py` + `QuestionResult`，在现有 Recall@20 / 引用准确率之外增加：

| 指标 | 定义 | 用途 |
|---|---|---|
| `first_pass_rate` | `attempts == 1` 且无违规的比例 | Step 1/2 主指标 |
| `quote_verbatim_rate` | 首轮 `『』` 内容逐字命中的比例 | Step 1 主指标 |
| `violation_counts` | 按 `no_citation` / `book_not_found` / `juan_not_found` / `hallucinated_link` / `bad_quote` 分类计数 | 定位病灶 |
| `avg_attempts` | 平均尝试轮数 | 延迟代价 |
| `refusal_rate` | 拆成 `rejected_by_threshold`（检索侧）与 `citation_failure`（生成侧）两项 | 区分"该拒答"和"不该拒答" |
| `prompt_tokens_p50 / p95 / max` | 见 0.1 | Step 1 定 ctx |
| `explain_score` | qwen-max 盲评释义质量 1–5 分，见 0.3 | Step 1/2/3 的最终指标 |

写进 `eval/report/<label>.json`，并让 `guji eval-compare` 能并排显示这些新列。

### 0.3 加释义质量裁判（新增 `guji eval --judge`）

- 对每题的最终回答，把「问题 + 检索到的原文段落 + 模型回答」交给 `qwen-max` 打 1–5 分，
  评分维度只有一个：**释义是否准确且忠于所引原文**（不评文采、不评格式，格式已由前面
  的机械指标覆盖）。
- 裁判必须**盲评**：不告诉它回答来自哪个模型。
- 默认关闭（要花云端 token），只在 A/B 时开。
- 判分 prompt 与温度固定写死在代码里，任何时候不得改动——改了历史 label 就不可比。

### 0.4 打开校验开关

`config.yaml` 当前是 `generate.validate_citations: false`。**所有 A/B 必须在
`true` 下跑**，否则测的是"没有约束的模型"，跟线上行为不一致。

### 验收标准

- `guji eval --label local_baseline --judge` 能在 `profile: local` 下跑完 80 题并产出
  上表全部字段。
- 同时跑一次 `--label cloud_baseline`（`profile: cloud`，qwen-max）作为天花板参照。
- `guji eval-compare local_baseline cloud_baseline` 输出对照表。

**这一步的产出是后面三步唯一的判据，不要跳过、不要边改功能边测量。**

---

## Step 1 — 调 ctx-size 与 KV cache 精度（改配置，不改代码）

### 依据

`config.yaml` 现在是 `--ctx-size 32768 --cache-type-k q4_0 --cache-type-v q4_0`。
K cache 量化到 4bit 对**精确复制类任务**的伤害显著大于对普通对话的伤害，而逐字回显
恰好是最敏感的一类。K 比 V 敏感得多，所以只升 K。

Qwen3-8B 32k 上下文的 KV 占用粗算：`q4_0/q4_0` ≈ 1.4 GB，`q8_0/q4_0` ≈ 2.0 GB，
`f16/f16` ≈ 4.8 GB。升 K 的代价约 +0.6 GB —— 而这笔钱从 ctx-size 里省回来。

### 做法

1. 从 Step 0 拿到 `prompt_tokens_p95` 与 `max`。**按 max × 1.5 向上取到 2 的幂**定
   新 ctx（预期落在 12288–16384；worst case 来自 `max_tool_rounds: 4` 连续召回叠加重试，
   不要凭直觉拍 8192）。
2. 三组 A/B，只改 `providers.local.llm.launch.command`：

   | label | 配置 |
   |---|---|
   | `local_baseline` | Step 0 已有：`ctx 32768, k=q4_0, v=q4_0` |
   | `kv_k8` | `ctx <新值>, k=q8_0, v=q4_0` |
   | `kv_f16` | `ctx <新值>, k=f16, v=f16`（上界参照，确认"KV 量化到底伤了多少"） |

3. 记录每组的**峰值内存**（`ask` 全程，M3 上用 `footprint` 或活动监视器）与延迟 P50/P95。

### 验收标准

- 若 `kv_k8` 相对 `local_baseline` 的 `quote_verbatim_rate` 提升 ≥ 5 个百分点 → 采纳，
  写回 `config.yaml`。
- 若 `kv_k8` 与 `kv_f16` 都无明显提升（< 2 个百分点）→ 结论是"KV 量化不是瓶颈"，
  记录到报告里，保留 `q4_0` 省内存，直接进 Step 2。
- 无论结果如何，**降低后的 ctx-size 都应采纳**（省内存且无损）。

> ⚠️ 这一步必须在 Step 2 之前单独做完。Step 2 会把逐字通过率推到 ~100%，
> 之后就永远无法测出 KV 量化原本伤了多少。

---

## Step 2 — 模糊锚点修复（核心改动，只动校验层）

### 设计

**不改输出协议**——模型照常输出 `『原文』`。改的是 `_validate`：当引文不是任何检索
段落的精确子串时，先尝试**对齐修复**，而不是立刻判违规。

修复成功 → 把回答里那段引文**替换成语料中的精确原文**，视为通过；
修复失败 → 走现有的拒绝-重试路径，行为与今天完全一致。

这样做的理由：模型的错通常是简繁漂移、脱字、衍字、标点差异，语义定位其实是对的。
把"精确复制"从模型手里拿走，交给代码。

### 实现要点（`src/guji/generate.py`）

- 新函数 `_repair_quote(qn: str, pool_norm: list[str]) -> tuple[str, float] | None`：
  - 对每个 passage 用 `difflib.SequenceMatcher(None, qn, p)` 找最长匹配块，
    由 opcodes 反推 `p` 上覆盖该引文的区间，得到候选修复串与相似度
    `ratio = 匹配长度 / len(qn)`。
  - 取全局最优候选。
- 接受条件（**两个都要满足**，防止把张冠李戴的引文"修"成另一段真原文）：
  - `ratio >= cfg.generate.quote_repair.min_ratio`（默认 `0.75`）
  - 修复串长度落在 `len(qn)` 的 0.8–1.25 倍之间
- 修复必须返回**语料原文的原始形态**（`text_raw` 上的切片，不是 `_norm_ws` 之后的），
  否则会把空白归一化的结果写进回答。
- `_validate` 返回值增加 `repairs: list[tuple[str, str]]`（原引文 → 修复后）；
  `answer()` 在判定通过后按 `repairs` 对 `text` 做替换，替换后的文本才是返回值。
- `AnswerResult` 增加 `quote_repairs: list[tuple[str, str, float]]`，
  eval 统计 `repair_rate`。**修复率高本身是个信号**（说明模型漂移严重），要能看到。

### 配置

```yaml
generate:
  quote_repair:
    enabled: true        # false 时行为与今天完全一致，作为 A/B 对照
    min_ratio: 0.75
```

### 测试（`tests/test_generate.py`）

至少覆盖：

1. 简繁漂移（`『克己复礼』` vs 语料 `克己復禮`）→ 修复成功
2. 脱字（漏一个字）→ 修复成功
3. 衍字（多插一个解释性的字）→ 修复成功
4. **完全编造的引文** → 不修复，仍判违规（这是最重要的一条，防止修复机制变成杜撰漏斗）
5. 引文横跨两个 passage 拼接而成 → 不修复（长度比或 ratio 卡住）
6. `enabled: false` 时行为与旧逻辑逐字一致

### 验收标准

- `guji eval --label repair --judge` 对比 Step 1 的最佳 label：
  - `quote_verbatim_rate`（含修复）→ 期望 ≥ 95%
  - `avg_attempts` 明显下降
  - `explain_score` **不得下降**（若下降，说明修复引入了错误对齐，检查接受条件）
- 测试 4 必须绿。

---

## Step 3 — 句级索引 + grammar 约束（**条件触发，非必做**）

**只有在 Step 2 之后 `first_pass_rate` 仍 < 85% 时才做。** 若已达标，停在 Step 2。

### 做法

- 代码按 `。！？；` 预切每条检索结果，把带编号的句子一起放进工具结果 JSON：
  `{"chunk_id": "...", "sentences": [{"s": 1, "text": "..."}, ...]}`
- 模型引用时输出 `『q:3.s2』`，代码回填。选择空间从"任意字符串"降为"十几个整数"。
- 用 llama-server 的 GBNF / `json_schema` 卡死输出结构，格式违规从"重试修正"变为
  "不可能发生"。
- 结构化引用同理：模型只给 `chunk_id`，`《書名》卷X‧篇名` 由代码从 meta 渲染，
  届时 `generate.py:200` 附近那段 `juan_not_found` 模糊匹配逻辑可以整段删除。

### 已知代价（做之前想清楚）

- CLI 的流式绿色渲染逻辑要改（引用在 `』` 闭合后才能展开）。
- 这是模型没见过的新输出格式，需要配 2–3 个 few-shot 示例，否则小模型会写出畸形的 ref。
- 与 Step 2 的修复机制并存时，两条路径都要保留（模型可能混用两种写法）。

---

## 明确的非目标（本 plan 不做）

- ❌ 蒸馏 / LoRA / 任何训练——那是另一条独立路线，且必须在本 plan 跑完、确认"零训练
  改动已经榨干"之后才评估
- ❌ 接入荀子等古籍专用模型做翻译子 agent——独立提案，前提假设尚未验证
- ❌ 字符偏移式引用（`『q:3#12-45』`）——让 LLM 数字符位置不可靠，已被句级索引取代
- ❌ 改检索链路（HyDE / rerank / chunk 策略）——本 plan 只动生成层，检索侧变量必须冻结，
  否则 A/B 不可比

## 全程铁律

1. **一次只改一个变量。** 每个 label 对应一份完整的 `config.yaml` 快照（`eval` 已有
   snapshot 机制，确认它记录了 launch command）。
2. **所有 A/B 在 `validate_citations: true` 下跑。**
3. **`profile: local` 是被测对象，`profile: cloud` 是天花板参照**，两者的 eval 都要留档。
4. 每步结束后把结论（含"没有效果"的结论）追加到本文件末尾的「实验记录」小节，
   下一步的决策依赖它。

## 实验记录

### Step 0 — 实现完成（2026-08-03），尚未跑 eval

代码改动：
- `src/guji/generate.py`：`_converse` 改为 `stream_options={"include_usage": True}`
  并返回 `(content, prompt_tokens_max)`；`AnswerResult` 新增 `prompt_tokens_max`、
  `quote_stats`（首轮 `{"total", "verbatim_ok"}`，诊断用，不受重试影响）。
- `src/guji/eval.py`：`QuestionResult` 新增 `attempts` / `prompt_tokens_max` /
  `quote_total` / `quote_verbatim_ok` / `violation_reasons`（首轮）/ `judge_score`；
  `summarize()` 新增 `generation` 分区（`first_pass_rate` / `quote_verbatim_rate` /
  `violation_counts` / `avg_attempts` / `rejected_by_threshold_rate` /
  `citation_failure_rate` / `prompt_tokens_p50,p95,max` / `explain_score`）；新增
  `judge_answer()`：盲评（不告知模型来源），固定走 `providers.cloud`（qwen-max），
  prompt/温度=0 硬编码，不可事后修改。
- `src/guji/cli.py`：`guji eval` 新增 `--judge` flag（默认关闭）；report 的
  `config` snapshot 补齐 `profile` / `llm_model` / `llm_base_url` /
  `llm_launch_command` / `validate_citations`，不再只记录检索侧几个字段；
  `guji eval` 与 `guji eval-compare` 都打印新增的 generation 诊断表。

**对计划文本的修正**（探索代码后发现的偏差，已按实际代码调整实现，未改动本文件以上正文）：
- 0.4 所说"当前 `validate_citations: false`"不准确——`config.yaml:126` 已是
  `true`，`config.py` 默认值也是 `true`。这一步是确认性的，不是改动性的。
- `_validate` 实际函数名是 `validate`（无下划线前缀）。
- `_converse` 原本 `stream=True` 且未开 `usage`，因此 `prompt_tokens` 在改动前完全
  拿不到，必须显式加 `stream_options={"include_usage": True}` 才能满足 0.1 的前提；
  这依赖 llama-server 支持该参数（近期 llama.cpp 版本支持，未在本机验证）。
- `guji eval` 原有的 "snapshot" 机制只存了几个检索调参字段（top_k/hyde/rerank/…），
  并不含 profile、模型、launch command——铁律 1 说"snapshot 机制已确认记录 launch
  command"是错的，已在这次改动里把 snapshot 扩展到包含这些字段。

**尚未做**：还没有实际跑 `guji eval --label local_baseline --judge` /
`cloud_baseline`，Step 0 的验收标准（80 题跑完、字段齐全、两份报告能被
`eval-compare` 对照）还未验证。下一步应该先跑通这两次 eval，确认
`stream_options.include_usage` 在本地 llama-server 上确实返回 usage（如果不支持，
`prompt_tokens_max` 会一直是 0，需要另想办法，例如非流式调用一次性拿 usage，或退化
成用 tokenizer 估算）。

<!-- 每步做完后在这里追加：label / 配置 / 关键指标 / 结论（采纳 or 放弃） -->
