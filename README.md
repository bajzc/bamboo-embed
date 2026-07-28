# guji-rag — 中文古籍 RAG 检索系统

离线可用的中文古籍问答与检索系统：现代汉语提问，检索繁体文言原文，生成带精确出处
（书名·卷·篇）的可溯源回答。

> **状态：Phase 1–5 全部完成**（语料解析 / 索引 / 检索链路 / 工具层与生成 / 评测）。

> **后端：Ollama**（跨平台，替代 spec 的 MLX）。同一 Ollama 模型在 Linux 开发机与
> macOS 部署机产出相同向量，索引无需重建。embedding 模型 `qwen3-embedding:0.6b`（1024 维）。
> LLM（HyDE / 生成）与 rerank 开发期走阿里云百炼（`qwen-max` / `qwen3-rerank`），
> 密钥放 `.env`（见 `.env.example`，已 gitignore）；部署可切本地。

## 环境

开发/运行通过 Nix flake + direnv + uv（纯解析阶段与平台无关，无需 MLX）：

```bash
direnv allow          # 首次：加载 flake devShell 并 `uv sync`
# 或手动： nix develop --command uv sync
```

flake 提供 `python311` + `uv` + `ollama-rocm`（AMD GPU 加速，非 `pkgs.ollama` 的纯 CPU 版）+
`llama-cpp`（Vulkan 后端），并设置 `LD_LIBRARY_PATH` 使 manylinux 轮子（`opencc` 等）在
NixOS 上可加载，以及 `HSA_OVERRIDE_GFX_VERSION` / `ROCR_VISIBLE_DEVICES` 供 ROCm 识别不在
官方支持列表内的 GPU（如 gfx1032 → 伪装为 gfx1030）并选定独显而非核显。所有模型/路径配置
只从 `config.yaml` 读取。若无 AMD GPU，把 flake 里的 `ollama-rocm` 换回 `ollama`（纯 CPU）
或 `ollama-cuda`（NVIDIA）。

embedding 固定走 **Ollama**（`qwen3-embedding:0.6b`，见下）；`local` profile 的 LLM
（HyDE + `guji ask` 生成）走 **llama-server**（Vulkan 后端，`llama-cpp.override { vulkanSupport
= true; }`）——Vulkan 靠系统级 mesa RADV 自动识别显卡，不需要 ROCm 那套 gfx 版本伪装。

```bash
# 下载模型（一次性，~18GB，IQ4_NL 量化）：
mkdir -p models
# 从 https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF （或 ModelScope 镜像）
# 下载 Qwen3.6-35B-A3B-UD-IQ4_NL.gguf 到 models/

# 启动 llama-server（GPU 卸载 + MoE 专家 CPU 卸载，--n-cpu-moe 按显存占用率现场调）：
llama-server \
  --host 127.0.0.1 --port 8080 \
  --model models/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf \
  --jinja \
  --device Vulkan0 \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --n-cpu-moe 26 \
  --flash-attn auto \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --cont-batching \
  --reasoning off
```

（`--reasoning off` 取代了文中教程的 `--chat-template-kwargs '{"enable_thinking":false}'`——
这台机器上装的 llama-server 版本已把后者标记为 deprecated。）

`--n-cpu-moe 26` 是在这台机器上实测调出来的：从 `20` 开始时显存占用率跑到 99.7%
（10.69/10.72GB，请求中途用 `rocm-smi --showmeminfo vram --showuse` 采样确认），几乎
没有 KV cache 空间；调到 `26` 后降到 78.4%（8.40/10.72GB），生成速度从 31.9 t/s 降到
24.5 t/s（`/v1/chat/completions` 响应的 `timings.predicted_per_second`）——仍明显快于
教程原文 8GB 显存下的 15-20 t/s。换了模型/量化/context 长度后请重新按这个方法调一次。

`--jinja` 是硬性要求（不是文中教程原始命令的一部分）：Phase 4 的 `search_passages` /
`lookup_char` / `get_context` function calling依赖 GGUF 内置的 chat template，缺了
`--jinja` 工具调用会静默失效。`--device Vulkan0` 同样是硬性要求——`llama-server
--list-devices` 在这台机器上会列出两个 Vulkan 设备（`Vulkan0` = 独显 RX 6750 GRE，
`Vulkan1` = 7900X 核显，通过 RADV 也暴露成了一个可用设备），不显式指定会有卸载分散
到核显上拖慢速度的风险。`--n-cpu-moe` 没有固定值——从高往低试，边跑请求边用
`rocm-smi`/`nvtop` 看显存占用，调到 ~80% 左右（留出 KV cache 空间）；这台机器 10GB
显存比教程原文的 8GB 宽裕，需要 CPU 卸载的专家层数应该比教程的 `33` 少。

## Phase 1 用法

```bash
guji fetch              # 1. clone 语料 -> data/raw/chtxt
guji manifest           # 2. 扫描 65 文件 -> data/manifest.json（应含 56 部非空书目）
guji normalize --check  # 3. 统计 PUA/扩展汉字 -> char_report.json + pua_map.json 骨架
guji parse              # 4. 解析 -> passages.jsonl（叙事+诗词） + dict.sqlite（字书）
guji stats              # 5. 各部类 chunk/字数分布 + 字数预算核对
guji verify             # 6. 验收：56 书、字数预算 ±2%、无 PUA 残留、随机抽样
```

`guji -v <cmd>` 开启 DEBUG 日志。

## Phase 2 用法（索引 + 检索）

需要本地 Ollama 服务与 embedding 模型：

```bash
ollama serve &                       # flake 已提供 ollama
ollama pull qwen3-embedding:0.6b
guji embed                           # 全量嵌入 -> data/lancedb（断点续跑；--book/--limit 可选）
guji index                           # 建 ANN 向量索引 + FTS5 字符 bigram 索引
guji search "克己復禮" --no-rerank    # 混合检索（RRF），返回带出处的段落
```

`guji embed` 可断点续跑（已嵌入的 chunk 会跳过），M3 上全量约 2–5 小时；开发机可用
`--book <id>` 或 `--limit N` 只嵌入子集验证。

## Phase 3 用法（检索链路）

完整链路：`query → dense(HyDE 假想文言文) + sparse(bigram) → RRF → rerank(qwen3-rerank)
→ 分数阈值 → 关联书去重`。

```bash
cp .env.example .env          # 填入 DASHSCOPE_API_KEY（HyDE LLM + rerank 用）
guji search "克己復禮 是什么意思" --hyde --debug          # --debug 显示各路召回 + HyDE 文本
guji search "..." --book 史記 --dynasty 漢 --category 史書  # 元数据过滤
guji search "..." --no-rerank                            # 关掉 rerank（走 RRF 顺序）
```

- rerank 最高分低于 `rerank.threshold`（默认 0.35）时返回「未檢索到相關記載」。
- `related_to` 书目去重：并行版本（`史記` / `史記三家注`）同一卷/篇的段落会折叠
  （三家注被 集解/索隱/正義 注文主导，文本相似度失效，故以「关联书 + 同卷篇」为准）；
  其余近重复走字符 bigram Jaccard ≥ 0.7。同书内不同段落不折叠。
- HyDE 默认开（`config.yaml: hyde.enabled`），`--no-hyde` 关闭做 A/B。

## Phase 4 用法（工具层与生成）

LLM 通过 function calling 使用三个工具（`src/guji/tools.py`）：

| 工具 | 作用 |
|---|---|
| `search_passages` | 混合检索叙事文本（复用 Phase 3 的 `hybrid.search`） |
| `lookup_char` | 精确查字书（`dict.sqlite`），字词训诂类问题优先于向量检索 |
| `get_context` | 沿 `prev_id`/`next_id` 链取某段的前后文 |

```bash
guji ask "「敝」在《說文》中如何解？"
guji ask "克己復禮這句話出自哪里？完整地引用原文" --debug   # --debug 显示工具调用轨迹 + 重试次数
```

三条硬约束，**代码层校验，不只是提示词**（`src/guji/generate.py`）：

1. **结构化引用**：回答中 `《書名》卷/篇名` 的书名与卷名必须真实存在于本轮已检索到的
   段落（`search_passages`/`get_context`/`lookup_char` 任一工具的结果）中，书名比较做
   了简繁与常见简称归一（如「說文」≈「說文解字」）；不存在则拒绝该回答，重试一次
   （`generate.retry_on_violation`），仍不通过则返回安全拒答，绝不放行编造的出处。
2. **逐字回显**：原文引用须用「`『』`」（`generate.quote_open/quote_close`）包裹，且
   与检索结果的 `text_raw` 逐字一致（允许空白差异），不允许改写、增删、简化；CLI 中
   模型的话与原文引用视觉分离（原文渲染为绿色 blockquote）。
3. **阈值门禁**：复用 Phase 3 的 rerank 分数阈值——最高分低于阈值时直接返回
   「未檢索到相關記載」，**不调用生成 LLM**（HyDE 仍会跑，因为它属于检索链路而非生成）。

## Phase 5 用法（评测）

`eval/questions.jsonl`：80 条，四类（字词训诂20 / 人物事件30 / 典故出处20 / 跨书比较10）。
每条附 `gold_books`（书名白名单）+ `gold_keywords`（原文必现关键词），出题前已逐条核对
`manifest.json` 白名单与实际语料（`data/passages.jsonl` / `dict.sqlite`），确保库里没有的书
（左傳、資治通鑑、紅樓夢等）不会被拿来出题。

```bash
guji eval --label baseline              # Recall@20 + 引用准确率，全 80 题
guji eval --label no_rerank --no-rerank  # A/B：关闭 rerank
guji eval --label no_hyde --no-hyde      # A/B：关闭 HyDE
guji eval-compare baseline no_rerank no_hyde   # 对比表
```

只看两个指标：
- **Recall@20**（检索层）：top-20 融合/重排结果中，是否有一条来自 `gold_books` 且包含
  `gold_keywords`。**字词训诂类例外**——字書不进向量库（设计如此），走 `hybrid.search`
  必然 0%，故该类改为直接查 `lookup_char`/`dict.sqlite` 是否命中，成本为零（无需
  LLM/rerank）。
- **引用准确率**（生成层）：`guji ask` 最终通过校验的回答，是否引用了 `gold_books`
  中至少一本书；被阈值门禁或引用校验拒答的一律记为不准确。

结果与逐题明细写入 `eval/report/<label>.json`，支持跑不同 config 做 A/B。

## 产物

| 文件 | 内容 |
|---|---|
| `data/manifest.json` | 书目元数据（书名/朝代/作者/部类/form/关联/缺口） |
| `data/passages.jsonl` | 叙事 + 诗词切块（进向量库），双字段 `text_raw`/`text_norm` |
| `data/dict.sqlite` | 字书条目（`char_entry`，**不进向量库**，供 `lookup_char` 工具） |
| `data/char_report.json` | PUA / BMP 外汉字统计 |
| `data/pua_map.json` | PUA 映射表（骨架为 `□`，可人工补充） |
| `data/lancedb/` | LanceDB 向量库（`qwen3-embedding:0.6b`，1024 维，cosine） |
| `data/fts.sqlite` | FTS5 字符 bigram 稀疏索引 + `meta` 展示字段 |

## 切分规则（§5.4）

| 形态 | 规则 |
|---|---|
| 叙事 | `## 卷/篇` 硬边界；段内合并至 250–400 字，重叠 1 段 |
| 诗词 | 一首一条（`○` 分隔），不切；无 `○` 的蒙学回退到叙事切法 |
| 辞书 | 一字头一条 → SQLite；康熙字典 p1+p2 合并 |

## 测试

```bash
pytest            # 各 parser ≥3 用例，覆盖点名的怪异格式文件
```

## 已知语料事实 / 缺口

- 全库 27,865,176 字，65 文件；`史書` 66% + `字書訓詁` 26%。
- `0.詩詞/` 8 个文件为空；`c.法家 e.墨家 f.雜家 k.醫學` 仅 `.gitkeep`。
- `康熙字典_p1/p2` 同书分片（合并）；`史記` 与 `史記三家注` 内容重叠（`related_to`，
  检索层去重待 Phase 3）。
- `a.儒家/詩經_symbolic.txt` 为指针桩（`內容見 0.詩詞/…`），标记 `is_stub`，不进 passages。
- 全库繁体；PUA 私有区字符统一映射（未命中 → `□`）。
