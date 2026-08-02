# guji-rag — 中文古籍 RAG 检索系统

离线可用的中文古籍问答与检索系统：现代汉语提问，检索繁体文言原文，生成带精确出处
（书名·卷·篇）的可溯源回答。

> **后端**：embedding 恒走本地 **Ollama**（`qwen3-embedding:0.6b`，1024 维，与
> `profile` 无关，跨平台产出相同向量，索引无需重建）。LLM（HyDE + `guji ask` 生成）
> 由 `config.yaml: profile` 切换——`local` 走本地 **llama-server**（GPU 后端由 Nix
> flake 按平台自动选：Linux 用 Vulkan，macOS 用 Metal）；`cloud` 走阿里云百炼
> `qwen-max`。rerank 默认走阿里云百炼 `qwen3-rerank`（`config.yaml: rerank.backend`
> 设为 `dashscope`），密钥放 `.env`（见 `.env.example`，已 gitignore）；不想依赖
> DashScope key，设为 `local` 改走另一个独立的 **llama-server --reranking** 进程
> （模型用 `guji models fetch reranker` 下载，见下方「本地模型（llama-server）」）。

## 快速开始

前提：已完成一次性搭建（语料抓取/解析/建索引，见下方「环境搭建」「语料处理」「建索引」），
即 `data/lancedb/` 与 `data/fts.sqlite` 已存在。这一步很慢（全量 embed 单机 1–5 小时），
如果有人已经发布过预建产物，直接下载会快得多——见下方「分发预建数据」。日常使用只需三步：
起 embedding 服务、起本地 LLM（若 `profile: local`）、问问题。

```bash
# 0. 进入开发环境（首次会自动 uv sync）
# Linux（Nix flake）：
direnv allow                    # 或: nix develop --command <cmd>
# macOS（Homebrew，见下方「环境搭建 › macOS（Homebrew）」）：
uv sync
```

**1) 启动 embedding 服务**（Ollama，`guji search`/`guji ask` 检索阶段必需，与 profile 无关）：

```bash
ollama serve &
ollama pull qwen3-embedding:0.6b   # 仅需一次
```

**2) 启动本地 LLM**（`config.yaml: profile: local` 时，HyDE 与 `guji ask` 生成都走它；
模型下载与参数调优见下方「环境搭建」）：

```bash
# Linux（Vulkan，见「环境搭建 › Linux」的 --n-cpu-moe 调参说明）
llama-server \
  --host 127.0.0.1 --port 8080 \
  --model models/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf \
  --jinja --device Vulkan0 --ctx-size 32768 \
  --n-gpu-layers 99 --n-cpu-moe 26 \
  --flash-attn auto --cache-type-k q4_0 --cache-type-v q4_0 \
  --cont-batching --reasoning off
```

```bash
# macOS（Metal，Apple Silicon；见「环境搭建 › macOS」）
llama-server \
  --host 127.0.0.1 --port 8080 \
  --model models/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf \
  --jinja --ctx-size 32768 \
  --n-gpu-layers 99 \
  --flash-attn auto --cache-type-k q4_0 --cache-type-v q4_0 \
  --cont-batching --reasoning off
```

若 `profile: cloud`，跳过这一步，改为在 `.env` 中填 `DASHSCOPE_API_KEY`（cloud LLM 走
`qwen-max`）。默认配置下 rerank 也走这把 key（`qwen3-rerank`，DashScope 不是 OpenAI
兼容接口，走独立 adapter）：

```bash
cp .env.example .env   # 填入 DASHSCOPE_API_KEY
```

不想依赖 DashScope：要么都加 `--no-rerank`（跳过 rerank，检索质量会下降，RRF 排序
直接生效）；要么把 `config.yaml: rerank.backend` 设为 `local`，起第三个 llama-server
进程做 reranking（模型 `guji models fetch reranker` 下载，独立端口，与 LLM 的
llama-server 互不影响）：

```bash
llama-server \
  --host 127.0.0.1 --port 8081 \
  --model models/Qwen3-Reranker-0.6B.Q8_0.gguf \
  --reranking --pooling rank --embedding
```

**3) 提问 / 检索**：

```bash
guji ask "「敝」在《說文》中如何解？"
guji ask "克己復禮這句話出自哪里？完整地引用原文" --debug   # 显示 HyDE/检索/工具调用/校验轨迹
guji ask "..." --hyde --validate                          # 开 HyDE 假想文言文 + 引用校验（拒绝并重试一次）
guji ask "..." --no-rerank                                 # 跳过 rerank，省 DashScope key
guji ask "..." --book 史記 --dynasty 漢 --category 史書    # 元数据过滤
```

`--hyde/--no-hyde`、`--validate/--no-validate`、`--rerank/--no-rerank` 不传时读
`config.yaml`（`hyde.enabled` / `generate.validate_citations` 当前均默认 `false`）。
回答流式输出；`--debug` 额外打印 HyDE 文本、检索命中、工具调用与每次重试的校验结果。

只检索不生成（更快，不占 LLM）：

```bash
guji search "克己復禮" --no-rerank
guji search "克己復禮 是什么意思" --hyde --debug   # --debug 显示各路召回 + HyDE 文本
```

### 内存占用：按需启停本地 llama-server（避免 16G OOM）

上面「快速开始」第 2 步是手动起两个常驻 llama-server（LLM + reranker），加上常驻的
embedding，三者同时占内存——在 16G 统一内存机器上很容易 OOM。可选：在
`config.yaml` 给 `providers.local.llm` / `rerank` 各加一个 `launch:` 块（示例已写在
`config.yaml` 里，注释掉了），guji 就会在真正需要时（`guji ask`/`guji search`/
`guji eval`）按需拉起对应的 `llama-server` 子进程，命令跑完就把自己启动的那个停掉；
如果 base_url 已经有服务在监听（比如你还是手动起的），guji 不会碰它，也不会去停它。
纯 opt-in：不加 `launch:` 就是原来的行为，一切不变。日志写在 `.guji/logs/`。

## 环境搭建

开发/运行分平台（纯解析阶段与平台无关，无需 MLX）：Linux 用 Nix flake + direnv + uv；
macOS 用 Homebrew + uv（不需要装 Nix）。所有模型/路径配置只从 `config.yaml` 读取，与
平台无关。

```bash
# Linux：
direnv allow          # 首次：加载 flake devShell 并 `uv sync`
# 或手动： nix develop --command uv sync

# macOS：见下方「macOS（Homebrew）」小节
```

### 本地模型（llama-server）

`guji models fetch` 从 Hugging Face 下载 GGUF，落到 `models/`（Linux/macOS 通用，同一批
文件）。可选 key：`llm-large`（35B-A3B MoE，~18GB，`profile: local` 主力模型）、
`llm-small`（8B dense，~4.8GB，16GB 统一内存机器如 M3 用这个）、`reranker`（0.6B
cross-encoder，配 `rerank.backend: local`）。`guji models list` 看完整目录（repo/文件/
说明）；不传参数等于全部下载：

```bash
guji models list                       # 看目录
guji models fetch llm-large            # 单个模型（一次性，~18GB）
guji models fetch llm-small reranker   # 也可以一次下多个
guji models fetch                      # 全部下载
```

`reranker` 用的是 [Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp](https://huggingface.co/Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp)，
用官方 `convert_hf_to_gguf.py` 转换——大多数社群转换版本缺 `cls.output.weight`
tensor，跑 `--reranking` 会得到近似 0 的分数，这个仓库的版本验证过可用。

### Linux（本项目开发机：AMD GPU）

flake 提供 `python311` + `uv` + `ollama-rocm`（AMD GPU 加速，非 `pkgs.ollama` 的纯 CPU 版）+
`llama-cpp.override { vulkanSupport = true; }`（Vulkan 后端），并设置 `LD_LIBRARY_PATH`
使 manylinux 轮子（`opencc` 等）在 NixOS 上可加载，以及 `HSA_OVERRIDE_GFX_VERSION` /
`ROCR_VISIBLE_DEVICES` 供 ROCm 识别不在官方支持列表内的 GPU（如 gfx1032 → 伪装为
gfx1030）并选定独显而非核显。Vulkan 靠系统级 mesa RADV 自动识别显卡，不需要 ROCm 那套
gfx 版本伪装。若无 AMD GPU，把 `flake.nix` 里 Linux 分支的 `pkgs.ollama-rocm` 换成
`pkgs.ollama`（纯 CPU）或 `pkgs.ollama-cuda`（NVIDIA）。

启动命令见「快速开始」第 2 步的 Linux 版本（GPU 卸载 + MoE 专家 CPU 卸载，`--n-cpu-moe`
按显存占用率现场调）。

（`--reasoning off` 取代了文中教程的 `--chat-template-kwargs '{"enable_thinking":false}'`——
这台机器上装的 llama-server 版本已把后者标记为 deprecated。）

`--n-cpu-moe 26` 是在这台机器上实测调出来的：从 `20` 开始时显存占用率跑到 99.7%
（10.69/10.72GB，请求中途用 `rocm-smi --showmeminfo vram --showuse` 采样确认），几乎
没有 KV cache 空间；调到 `26` 后降到 78.4%（8.40/10.72GB），生成速度从 31.9 t/s 降到
24.5 t/s（`/v1/chat/completions` 响应的 `timings.predicted_per_second`）——仍明显快于
教程原文 8GB 显存下的 15-20 t/s。换了模型/量化/context 长度后请重新按这个方法调一次。

`--jinja` 是硬性要求（不是文中教程原始命令的一部分）：问答生成用到的 `search_passages` /
`lookup_char` / `get_context` function calling依赖 GGUF 内置的 chat template，缺了
`--jinja` 工具调用会静默失效。`--device Vulkan0` 同样是硬性要求——`llama-server
--list-devices` 在这台机器上会列出两个 Vulkan 设备（`Vulkan0` = 独显 RX 6750 GRE，
`Vulkan1` = 7900X 核显，通过 RADV 也暴露成了一个可用设备），不显式指定会有卸载分散
到核显上拖慢速度的风险。`--n-cpu-moe` 没有固定值——从高往低试，边跑请求边用
`rocm-smi`/`nvtop` 看显存占用，调到 ~80% 左右（留出 KV cache 空间）；这台机器 10GB
显存比教程原文的 8GB 宽裕，需要 CPU 卸载的专家层数应该比教程的 `33` 少。

### macOS（Homebrew）

不需要装 Nix。直接用 Homebrew 装 Python 3.11、uv、Ollama、llama.cpp——Homebrew 的
`ollama` 与 `llama.cpp` bottle 在 macOS 上自带 Metal/Accelerate 加速，不需要额外
编译参数（对应 Linux 分支里 Nix flake 手动切换 `ollama-rocm`/`vulkanSupport` 的部分，
在 macOS 上是包本身默认就有的行为）：

```bash
brew install python@3.11 uv ollama llama.cpp
```

固定 uv 使用 Homebrew 装的 3.11 解释器，避免 uv 另外下载一份 Python：

```bash
export UV_PYTHON="$(brew --prefix python@3.11)/bin/python3.11"
export UV_PYTHON_DOWNLOADS=never
uv sync
```

（这两个环境变量写进 shell rc 长期生效，或者每次 `uv sync` 前手动 export 一次都行；
不需要 direnv，也不需要 `flake.nix`——那份 flake 只服务 Linux 开发机。）

llama-server 启动命令见「快速开始」第 2 步的 macOS 版本，与 Linux 版本的差异：
- 不需要 `--device`：Metal 由 llama-cpp 自动检测，不像 Vulkan 那样一台机器可能列出
  多个设备（独显/核显）需要手动选一个。
- 不建议默认加 `--n-cpu-moe`：Apple Silicon 是统一内存架构，GPU/CPU 共享同一块
  RAM，Linux 独显那套「专家卸载到 CPU 省 VRAM」的逻辑不成立——卸载省不下总内存
  占用，只是把计算挪到较慢的 CPU 上。模型放不进统一内存（OOM 或系统开始疯狂
  swap）时再考虑加，或者换更小的量化（如 IQ3 系列）。
- 调优看统一内存压力而非「VRAM」：没有 `rocm-smi`/`nvtop`，用「活动监视器」的
  内存压力表，或 `sudo powermetrics --samplers gpu_power` 观察 GPU 占用。

Intel Mac（`x86_64`）与 Apple Silicon（`arm64`）Homebrew 都提供原生 bottle，装法不变；
GPU 加速效果取决于机器实际的 Metal 支持情况（Apple Silicon 上稳定可用，Intel Mac 上
因显卡型号而异，跑不动就退化成 CPU 推理），其余步骤不变。

## 语料处理

```bash
guji fetch              # 1. clone 语料 -> data/raw/chtxt
guji manifest           # 2. 扫描 65 文件 -> data/manifest.json（应含 56 部非空书目）
guji normalize --check  # 3. 统计 PUA/扩展汉字 -> char_report.json + pua_map.json 骨架
guji parse              # 4. 解析 -> passages.jsonl（叙事+诗词） + dict.sqlite（字书）
guji stats              # 5. 各部类 chunk/字数分布 + 字数预算核对
guji verify             # 6. 验收：56 书、字数预算 ±2%、无 PUA 残留、随机抽样
```

`guji -v <cmd>` 开启 DEBUG 日志。

## 建索引

需要本地 Ollama 服务与 embedding 模型（见「快速开始」第 1 步）：

```bash
guji embed                           # 全量嵌入 -> data/lancedb（断点续跑；--book/--limit 可选）
guji index                           # 建 ANN 向量索引 + FTS5 字符 bigram 索引
guji search "克己復禮" --no-rerank    # 混合检索（RRF），返回带出处的段落
```

`guji embed` 可断点续跑（已嵌入的 chunk 会跳过），M3 上全量约 2–5 小时；开发机可用
`--book <id>` 或 `--limit N` 只嵌入子集验证。

## 分发预建数据

语料解析 + 建索引全程可能要几个小时，没必要让每个使用者都重跑一遍。`guji release`
把「产物」一节列出的可再生产物（`manifest.json`/`passages.jsonl`/`dict.sqlite`/
`char_report.json`/`pua_map.json`/`lancedb/`/`fts.sqlite`，不含 `data/raw/` 原始语料
——那只是上游仓库的一份 clone，用 `guji fetch` 重新拉取即可）打成一个 tar.gz，发布到
Hugging Face 数据集仓库，其他人下载解压后可以直接 `guji search`/`guji ask`，跳过
「语料处理」「建索引」两节。

```bash
uv sync --extra release   # huggingface_hub，仅打包/发布/下载需要，日常问答不需要
```

**打包**（本地操作，不需要网络/账号）：

```bash
guji release pack --label 20260801   # 默认 out-dir=dist/，label 默认取今天日期
# -> dist/guji-rag-data-20260801.tar.gz + dist/guji-rag-data-20260801.manifest.json
```

manifest 里记了 embedding 模型/维度、chunk 参数、git commit、整包与每个文件的 sha256，
供下载方在解压前核对兼容性与完整性。

**发布**（需要 `huggingface-cli login` 或 `HF_TOKEN` 环境变量，会创建/更新一个公开的
HF 数据集仓库——除非加 `--private`）：

```bash
guji release publish yourname/guji-rag-data   # 默认取 dist/ 下最新的包
```

**获取**（另一台机器/另一个使用者）：

```bash
guji release fetch yourname/guji-rag-data --label 20260801
```

下载后先校验整包 sha256，再检查本地 `config.yaml: embedding` 的模型/维度是否与打包时
一致（不一致会拒绝解压——向量空间对不上）；解压路径来自打包时的 `config.yaml: paths`，
落地位置与「产物」一节一致。`data/` 下已有同名产物时默认拒绝覆盖，加 `--force` 覆盖。

## 检索

完整链路：`query → dense(HyDE 假想文言文) + sparse(bigram) → RRF → rerank(qwen3-rerank)
→ 分数阈值 → 关联书去重`。

```bash
cp .env.example .env          # 填入 DASHSCOPE_API_KEY（rerank 必需；HyDE/生成走 cloud profile 时也用它）
guji search "克己復禮 是什么意思" --hyde --debug          # --debug 显示各路召回 + HyDE 文本
guji search "..." --book 史記 --dynasty 漢 --category 史書  # 元数据过滤
guji search "..." --no-rerank                            # 关掉 rerank（走 RRF 顺序）
```

- rerank 最高分低于 `rerank.threshold`（默认 0.35）时返回「未檢索到相關記載」。
- `related_to` 书目去重：并行版本（`史記` / `史記三家注`）同一卷/篇的段落会折叠
  （三家注被 集解/索隱/正義 注文主导，文本相似度失效，故以「关联书 + 同卷篇」为准）；
  其余近重复走字符 bigram Jaccard ≥ 0.7。同书内不同段落不折叠。
- HyDE 由 `config.yaml: hyde.enabled` 控制（当前默认关），`--hyde/--no-hyde` 覆盖做 A/B。

## 问答生成

LLM 通过 function calling 使用三个工具（`src/guji/tools.py`）：

| 工具 | 作用 |
|---|---|
| `search_passages` | 混合检索叙事文本（复用「检索」一节的 `hybrid.search`） |
| `lookup_char` | 精确查字书（`dict.sqlite`），字词训诂类问题优先于向量检索 |
| `get_context` | 沿 `prev_id`/`next_id` 链取某段的前后文 |

```bash
guji ask "「敝」在《說文》中如何解？"
guji ask "克己復禮這句話出自哪里？完整地引用原文" --debug   # --debug 显示工具调用轨迹 + 重试次数
```

回答流式输出（token 到达即打印，含重试轮次），原文引用实时渲染为绿色。模型自己的
解释、分析用简体中文书写；书名/卷篇与「`『』`」内的逐字引用保留检索结果的原始繁体，
不转换（系统提示词 `generate._SYSTEM_PROMPT` 规则 6）。

三条硬约束，**代码层校验，不只是提示词**（`src/guji/generate.py`），由
`config.yaml: generate.validate_citations`（当前默认 `false`）或 `ask --validate/--no-validate`
开关：

1. **结构化引用**：回答中 `《書名》卷/篇名` 的书名与卷名必须真实存在于本轮已检索到的
   段落（`search_passages`/`get_context`/`lookup_char` 任一工具的结果）中，书名比较做
   了简繁与常见简称归一（如「說文」≈「說文解字」）；不存在则拒绝该回答，重试一次
   （`generate.retry_on_violation`），仍不通过则返回安全拒答，绝不放行编造的出处。
2. **逐字回显**：原文引用须用「`『』`」（`generate.quote_open/quote_close`）包裹，且
   与检索结果的 `text_raw` 逐字一致（允许空白差异），不允许改写、增删、简化；CLI 中
   模型的话与原文引用视觉分离（原文渲染为绿色）。
3. **阈值门禁**：复用「检索」一节的 rerank 分数阈值——最高分低于阈值时直接返回
   「未檢索到相關記載」，**不调用生成 LLM**（HyDE 仍会跑，因为它属于检索链路而非生成）。

`--no-validate` 跳过以上校验与重试，直接返回模型第一次的回答（不保证可溯源）。

## 评测

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

## 切分规则

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
  检索层按上方「检索」一节的规则去重）。
- `a.儒家/詩經_symbolic.txt` 为指针桩（`內容見 0.詩詞/…`），标记 `is_stub`，不进 passages。
- 全库繁体；PUA 私有区字符统一映射（未命中 → `□`）。