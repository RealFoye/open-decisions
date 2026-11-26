# open-decisions

**自托管的开源 Decisions API：一次请求，对同一段输入回答上百道是非 / 选择 / 打分题，返回校准过的概率，而不是生成文本。**

*A self-hosted, open-source Decisions API. One request, hundreds of yes/no, multiple-choice and rating questions over one shared input, calibrated probabilities back instead of text.* [English below](#english)

```text
            ┌──────────── 共享前缀（只算一次）────────────┐
  请求 ──▶  [system + 输入文档 / 工单 / 对话 …]  ──prefill──▶ KV cache ──┐
                                                                      │ broadcast，不复制
     q1: 是否要求退款？        ─┐                                        ▼
     q2: 分到哪个队列？        ─┼─ batched ──▶  P(Yes) / P(A..D) / P(0..4)  ──▶ 校准 ──▶ answers
     …   q1000                ─┘
```

## 为什么做这个

2026 年 11 月 11 日谷歌发布了闭源的 Gemini Decisions API（模型 `gemini-decision-1`）：不生成文本，只给封闭集合的问题打概率，一次最多 1000 题，用于工具选择、策略检查、模型路由、内容审核这类“小决定”。这类需求在 Agent 系统里无处不在，但开源这边目前只有“调一下 API 读 logprobs”的薄封装：每道题一次完整调用，长输入被重复计算上百次，概率也没有校准。

open-decisions 把它做成一个可以自己部署的服务：

- **共享前缀 KV 复用**：输入只 prefill 一次，所有问题在同一份 KV cache 上批量打分；同一输入的后续请求直接命中前缀缓存。结果与逐题完整计算数值一致（测试里误差 < 1e-4；Qwen2.5-0.5B fp32 实测 7e-5），在 CPU 上 100 道题比逐题调用快 34.6 倍。
- **多 token 选项**：选项标签不止一个 token 时（如 `10`、`refund_request`），用 teacher forcing 算整段序列的对数概率，只对这些选项额外跑一轮。
- **概率校准**：温度缩放（所有题型）、Platt 缩放（是非题，顺便修正“爱说 Yes”的偏差）、无标注的 contextual calibration；附带 `opendecisions calibrate` 命令，输出校准前后的 ECE / NLL。
- **多后端 + 编排**：本地 Hugging Face 模型（默认，无需任何 key），以及可选的任意 OpenAI 兼容接口（OpenAI、vLLM、SGLang、Ollama…）和 Claude；可以串成 **cascade**（小模型先答，没把握的再升级给大模型）或 **ensemble**（多模型概率平均）。
- **Decisions 风格的接口**：`POST /v1beta/decisions`，请求是一个 `input` 加一组 `questions`，返回带 `confidence` 的类型化答案，和 Gemini Decisions 的形状接近，迁移成本低。

## 快速开始

```bash
git clone https://github.com/RealFoye/open-decisions.git && cd open-decisions
pip install -e ".[local,server]"        # CPU 用户可先装 CPU 版 torch；Claude 后端另装 ".[anthropic]"

opendecisions serve --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
curl -s localhost:8000/v1beta/decisions -H 'content-type: application/json' -d @examples/request.json
```

```python
from opendecisions import DecisionsClient

c = DecisionsClient("http://localhost:8000")
r = c.decide("我两周前买的台灯到货晚了五天，而且灯罩裂了，我要退款。", {
    "wants_refund": {"type": "noul", "instructions": "用户要求退款。"},
    "queue": {"type": "choice", "instructions": "该分到哪个客服队列？",
              "options": {"returns": "破损、退货退款", "shipping": "物流延误", "billing": "账单发票", "other": "其他"}},
    "urgency": {"type": "score", "instructions": "需要多快回复？",
                "levels": ["一周内", "三天内", "24 小时内", "一小时内"]},
})
r.answers["wants_refund"].noul      # P(真) ∈ [0.01, 0.99]
r.answers["queue"].probabilities    # {"billing": .., "other": .., "returns": .., "shipping": ..}，和为 1
r.answers["urgency"].score          # 按概率加权的等级期望值
```

不想起服务也可以进程内直接用：见 `examples/quickstart.py`。

### 请求与返回

| 题型 | 请求字段 | 返回字段 |
|---|---|---|
| `noul` 是非 | `instructions`，可选 `true_description` / `false_description` | `noul`（陈述为真的概率）、`confidence = max(p, 1-p)` |
| `choice` 单选 | `instructions`、`options: {key: 描述}`（2–255 个，按 UTF-8 字节序排序） | `choice`、`probabilities`、`confidence` |
| `score` 打分 | `instructions`、`levels: [等级 0 描述, 等级 1 描述, …]` | `score`（期望等级）、`probabilities`、`legend`、`confidence` |

每个答案还带 `method`（`logprob` / `sampling` / `verbalized`）、`estimated`（是否是估计值）、`backend`、`escalated`。`usage` 里有 `prefix_tokens`、`prefix_tokens_reused`、`prefix_cache_hit`，可以直接看到复用了多少计算。限制和 Gemini 一致：每次 1–1000 题，题目 key 匹配 `[A-Za-z0-9_-]{1,64}`。

## 后端

`local` 是默认后端，不需要任何 API key，下面所有真实模型的结果都来自它。`openai` 和 `anthropic` 是**可选**后端：

- 调用 OpenAI 需要 `OPENAI_API_KEY`（指向自己部署的 vLLM / Ollama 时可以不填，用 `OPENAI_BASE_URL` 或配置里的 `base_url` 指过去）；调用 Claude 需要 `pip install -e ".[anthropic]"` 和 `ANTHROPIC_API_KEY`。
- 这两个后端目前只用模拟的 HTTP 响应 / 假客户端做了单元测试（`tests/test_remote_backends.py`），**还没有在真实 API 上跑过**。不配置它们不影响其他功能。

| 后端 | 概率从哪来 | 共享前缀怎么复用 | 说明 |
|---|---|---|---|
| `local`（Hugging Face transformers） | 真实 token 对数概率，**精确** | 进程内 KV cache：prefill 一次，按批广播；跨请求 LRU 前缀缓存 | 支持 chat template 和纯文本两种 prompt |
| `openai`（可选，任意 OpenAI 兼容接口） | `logprobs` + `top_logprobs`（取前 20）。模型不返回 logprobs 时自动退回到一次请求采样 `n` 个答案，标记为 `estimated` | 共享部分放在消息最前面，命中 OpenAI 自动提示缓存或 vLLM / SGLang 的 prefix caching | 生产环境推荐：`vllm serve <model> --enable-prefix-caching`，再用 `examples/vllm.json` 指过去 |
| `anthropic`（可选，Claude，官方 SDK） | Claude 的 Messages API **不返回 logprobs**，只能估计：`sampling`（结构化输出把答案限定在选项里，采样 k 次，Dirichlet 平滑频率）或 `verbalized`（一次调用让模型给出每个选项的概率） | 输入作为单独的内容块并加 `cache_control`，第一题之后从提示缓存读 | 答案一律 `estimated=true`；verbalized 概率建议先校准再用 |

关于不支持 logprobs 的模型，这里的做法是**如实标注而不是假装精确**：估计出来的概率会带 `estimated: true` 和对应的 `method`，下游可以按需区别对待（比如在 cascade 里只把它放在最后一级）。

## 编排模式（Orchestrator）

```json
{
  "strategy": "cascade",
  "stages": [
    {"backend": "local", "model": "Qwen/Qwen2.5-1.5B-Instruct", "min_confidence": 0.85, "calibration": "calibration.json"},
    {"backend": "openai", "model": "gpt-4.1-mini", "min_confidence": 0.85},
    {"backend": "anthropic", "model": "claude-opus-5-5", "method": "sampling", "samples": 8}
  ]
}
```

```bash
opendecisions serve --config examples/cascade.json
```

上面的例子第二、三级要用到可选的 OpenAI / Claude 后端，需要对应的 key。没有 key 时每一级都可以是 `local`，比如小模型在前、大模型在后。

- **cascade**：第一级回答全部问题；校准后置信度低于 `min_confidence` 的题才升级到下一级，依此类推。大部分简单题在本地一次共享前缀推理里答完，只有少量难题花 API 的钱。每个答案标明由哪个后端给出、是否升级过。
- **ensemble**：每一级都回答所有题，按 `weight` 平均概率分布，适合当更稳的“评审团”。

阈值只在校准过的概率上才有意义，所以建议每一级先跑一次 `opendecisions calibrate`。

## 校准

准备一份 JSONL，每行一道带标准答案的题（同一 `input` 的题会合并成一次共享前缀推理）：

```json
{"input": "盘子碎了，请退款。", "question": {"type": "noul", "instructions": "用户要求退款。"}, "label": true}
```

```bash
opendecisions calibrate --model Qwen/Qwen2.5-1.5B-Instruct --data examples/calibration_data.jsonl -o calibration.json
opendecisions serve --model Qwen/Qwen2.5-1.5B-Instruct --calibration calibration.json
```

命令会留出 30% 做验证，打印校准前后的准确率、NLL 和 ECE。温度缩放不改变 argmax，所以不会影响准确率。

## 基准：逐题调用 vs 共享前缀

“逐题调用”就是把每道题当成一次独立的完整请求，这正是只包装 API 的做法在自托管模型上的成本。题型按是非 / 四选一 / 五级打分轮换，batch size 16。最后一列是两种算法的数值差，证明结果一致。

### 真实模型：Qwen2.5-0.5B-Instruct

`python benchmarks/bench_prefix.py --model Qwen/Qwen2.5-0.5B-Instruct`。fp32，Apple M4 Pro CPU（torch 8 线程），输入 1,500 token：

| 题数 questions | 逐题调用 naive | 共享前缀（冷） shared, cold | 共享前缀（热，命中前缀缓存） shared, warm | 加速（冷） | 实际计算的 token 数 | 最大 \|Δ log p\| |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 48.79 s | 3.09 s | 1.41 s | 15.8× | 39,679 → 2,719 | 4.1e-05 |
| 50 | 264.13 s | 7.63 s | 5.96 s | 34.6× | 206,495 → 7,835 | 6.2e-05 |
| 100 | 526.72 s | 15.22 s | 16.29 s | 34.6× | 420,979 → 14,419 | 7.0e-05 |

热缓存只省掉一次前缀 prefill（这里约 1–2 秒），题目一多，冷热就差不多了；100 题那一行热比冷慢 1 秒是计时波动。

**关于 dtype**：transformers 5 不指定 `dtype` 时按 checkpoint 的类型加载，Qwen2.5 是 bf16。CPU 上 bf16 比 fp32 慢约 3 倍，而且舍入误差会让两种算法的 log p 相差 0.28–0.45。所以本地后端在 CPU 上默认用 fp32，在 GPU / MPS 上默认沿用 checkpoint 的类型。`serve`、`calibrate` 和两个基准脚本都可以用 `--dtype` 改（`float32` / `bfloat16` / `float16` / `auto`），JSON 配置里写 `"dtype": "bfloat16"` 也行。

### 离线：随机权重模型

不加 `--model` 时不需要下载：用 19M 参数的随机权重 Llama 结构测速度（速度和权重无关）。4 核 CPU，torch 4 线程，输入 5,753 token：

| 题数 questions | 逐题调用 naive | 共享前缀（冷） shared, cold | 共享前缀（热） shared, warm | 加速（冷） | 实际计算的 token 数 | 最大 \|Δ log p\| |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 20.70 s | 3.03 s | 1.48 s | 6.8× | 76,387 → 6,679 | 1.1e-05 |
| 50 | 109.09 s | 8.31 s | 6.83 s | 13.1× | 393,777 → 10,383 | 2.3e-05 |
| 100 | 212.55 s | 16.35 s | 14.58 s | 13.0× | 781,687 → 14,899 | 2.3e-05 |

共享前缀之后，每道题只剩问题本身的 token 要算；剩下的开销主要是 transformers 里把前缀 KV 拼进每个批次的内存拷贝。vLLM / SGLang 这类分页注意力引擎（可选的 `openai` 后端 + prefix caching）应该能把这部分也省掉，但还没有实测。

## 准确率与校准：Qwen2.5-0.5B-Instruct

`python benchmarks/eval_accuracy.py --model Qwen/Qwen2.5-0.5B-Instruct --n 400`（另需 `pyarrow`）。每种题型用一个公开的 Hugging Face 数据集，随机抽 400 条（seed 0）：一半拟合校准器，另一半 200 条做测试。fp32，CPU，每条输入一道题。

| 题型 | 数据集 | 多数类基线 | 方法 | 准确率 | NLL | ECE |
|---|---|---:|---|---:|---:|---:|
| `noul` | SST-2（“这条影评是正面的”） | 0.555 | 原始 | 0.775 | 0.485 | 0.085 |
| | | | contextual（无需标注） | 0.730 | 0.648 | 0.156 |
| | | | 拟合（温度 + Platt，200 条标注） | **0.825** | **0.445** | **0.057** |
| `choice` | AG News（新闻四分类） | 0.305 | 原始 | 0.775 | 1.209 | 0.195 |
| | | | contextual | 0.775 | 1.275 | 0.191 |
| | | | 拟合（温度） | 0.775 | **0.668** | **0.084** |
| `score` | Yelp 星级（1–5 星） | 0.245 | 原始 | 0.230 | 2.751 | 0.471 |
| | | | contextual | **0.435** | 2.155 | 0.328 |
| | | | 拟合（温度） | 0.230 | **1.458** | **0.170** |

`score` 期望等级的平均绝对误差：原始 1.20，contextual 0.76，拟合 0.98。

- **是非题、单选题**：0.5B 模型已经明显好于多数类基线，拟合的校准器把 ECE 降了一半以上。是非题 Platt 缩放的偏置项为负（b = −0.70），说明模型偏爱回答 Yes；修正后准确率从 0.775 升到 0.825。
- **打分题**：原始准确率和瞎猜差不多，但问题不在排序，而在整体偏高。另外在同一批测试样本上单独查过：期望等级和真实星级的秩相关是 0.77，平均预测 3.24 级，真实平均 2.02 级。温度缩放只改变分布的尖锐程度，改不了这种偏移，所以准确率不变。contextual calibration 恰好去掉先验偏移，把准确率提到 0.435，但它在是非题上反而变差。**校准方式要按题型选**，不要一刀切。
- **速度**：打分题每条约 500 ms，是非 / 单选约 110 / 150 ms。原因是 Yelp 评论更长，而且 Qwen 分词器把 ` 0` 切成“空格 + 数字”两个 token，等级标签要多跑一轮 teacher forcing。
- 这是 0.5B 的小模型。这些数字说明管线在真实模型上跑通、校准确实起作用，不代表方法的上限。

## 和 Gemini Decisions API 的区别

- 开源、自托管，模型可换；数据不出机房。
- 目前只支持文本输入（Gemini 支持图片、音频、视频、PDF）。
- `choice` / `score` 用显式的 `options` / `levels` 字段，而不是 `criteria`。
- 概率质量取决于你选的模型和是否校准；仓库不声称和 `gemini-decision-1` 等效。

## 开发

```bash
pip install -e ".[dev]"
pytest -q          # 全部离线，CPU 上几秒跑完：用本地训练的小分词器 + 随机小模型
```

路线图：原生 vLLM 后端（直接用 prompt logprobs 做多 token 选项）、多模态输入、Claude Message Batches、按题型的自动阈值选择。

许可证：Apache-2.0。

---

<a id="english"></a>

## English

**open-decisions** is an open-source, self-hosted take on the "Decisions" pattern that Google shipped as the closed Gemini Decisions API on 11 Nov 2026: instead of generating text, the model scores closed-set questions — yes/no (`noul`), multiple choice (`choice`) and ordered rating (`score`) — over one shared input, up to 1,000 per request, and returns probabilities with a `confidence`. That is what agent systems need for tool selection, policy checks, routing, moderation and grading.

What it adds over thin "call the API and read logprobs" wrappers:

- **Shared-prefix KV reuse.** The input is prefilled once; every question is scored in batches on top of the same KV cache (broadcast with `expand`, no copy), and repeated inputs hit a cross-request prefix cache. Results match naive per-question scoring to < 1e-4 (tested).
- **Multi-token options.** Labels that span several tokens are scored by teacher forcing, in an extra pass for just those options.
- **Calibration.** Temperature scaling for every question type, Platt scaling for yes/no (which also removes the "yes" bias), label-free contextual calibration, and an `opendecisions calibrate` command that reports ECE / NLL before and after on a held-out split.
- **Backends and orchestration.** Local Hugging Face models (the default; no API key needed), plus optional backends for any OpenAI-compatible endpoint (OpenAI, vLLM, SGLang, Ollama, …) and Claude, composable as a **cascade** (a small model answers everything; low-confidence answers escalate to the next stage) or an **ensemble** (weighted average of distributions).
- **Decisions-style API.** `POST /v1beta/decisions` (and `/v1/decisions`) with an `input` and a `questions` map; typed answers with `confidence`, `method`, `estimated`, `backend` and `escalated`; `usage` shows how many prefix tokens were reused.

### Quick start

```bash
pip install -e ".[local,server]"
opendecisions serve --model Qwen/Qwen2.5-1.5B-Instruct
curl -s localhost:8000/v1beta/decisions -H 'content-type: application/json' -d @examples/request.json
```

### Where the probabilities come from

`local` is the default backend and needs no API key; every real-model result below comes from it. `openai` and `anthropic` are **optional**: they need `OPENAI_API_KEY` (not required for a self-hosted vLLM / Ollama endpoint set via `OPENAI_BASE_URL` or `base_url`) and `pip install -e ".[anthropic]"` plus `ANTHROPIC_API_KEY`. Both are unit-tested against mocked HTTP responses and a fake client only (`tests/test_remote_backends.py`) and **have not yet been run against the real APIs**. Nothing else depends on them.

| Backend | Probabilities | Prefix reuse |
|---|---|---|
| `local` | exact token log-probabilities | in-process KV cache, prefilled once, LRU across requests |
| `openai` (optional) | `logprobs`/`top_logprobs`; if the model returns none, falls back to sampling `n` answers in one request (`estimated: true`) | shared part first, so OpenAI automatic caching or vLLM/SGLang prefix caching kicks in |
| `anthropic` (optional) | the Messages API exposes no logprobs, so estimates only: `sampling` (answers constrained to the labels with structured outputs, smoothed frequencies) or `verbalized` (the model states a probability per label) — always `estimated: true` | the input is its own content block with `cache_control` |

Estimated probabilities are labelled as such rather than passed off as exact.

### Benchmark

Real model: `python benchmarks/bench_prefix.py --model Qwen/Qwen2.5-0.5B-Instruct`, fp32 on an Apple M4 Pro CPU (8 torch threads), 1,500-token input, questions alternating yes/no, 4-way choice and 5-level score. 100 questions take 526.7 s scored one by one and 15.2 s with a shared prefix (34.6×), computing 420,979 vs 14,419 tokens, with identical results (max |Δ log p| 7.0e-05). At 10 questions the speed-up is 15.8×, at 50 it is 34.6×.

transformers 5 loads the checkpoint dtype (bf16 for Qwen2.5) unless told otherwise. On CPU that is about 3× slower than fp32, and bf16 rounding makes the two algorithms differ by 0.28–0.45 in log p. The local backend therefore defaults to fp32 on CPU and to the checkpoint dtype on GPU / MPS. `serve`, `calibrate` and both benchmark scripts take `--dtype` (`float32` / `bfloat16` / `float16` / `auto`), and JSON configs accept `"dtype"`.

Without `--model` it runs offline on a 19M-parameter random-weight Llama (timings do not depend on weights): 4-core CPU, 5,753-token input, 100 questions in 212.6 s naively vs 16.4 s shared (13×), max |Δ log p| 2.3e-05. See the tables above. After sharing, most of the remaining cost is KV copying inside transformers. A paged-attention server (vLLM/SGLang via the optional `openai` backend) should remove that, but this has not been measured yet.

### Accuracy and calibration

`python benchmarks/eval_accuracy.py --model Qwen/Qwen2.5-0.5B-Instruct --n 400` (needs `pyarrow`) samples 400 examples per question type from public Hugging Face datasets (seed 0), fits the calibrator on one half and tests on the other 200. Results for Qwen2.5-0.5B-Instruct, fp32, CPU (full table in the Chinese section above):

- `noul`, SST-2 sentiment (majority baseline 0.555): accuracy 0.775 raw → 0.825 with fitted temperature + Platt scaling; ECE 0.085 → 0.057. The Platt bias is negative (b = −0.70), i.e. the model leans towards "Yes".
- `choice`, AG News 4-way topic (baseline 0.305): accuracy 0.775; temperature scaling cuts NLL from 1.209 to 0.668 and ECE from 0.195 to 0.084.
- `score`, Yelp 1–5 stars (baseline 0.245): raw accuracy is at chance (0.230), but because of a shift, not bad ranking. A separate check on the same test half gave rank correlation 0.77 between expected and true level, with predictions about 1.2 levels too high on average. Temperature scaling cannot remove a shift. Label-free contextual calibration does: accuracy 0.435, MAE 1.20 → 0.76. On yes/no questions, though, it makes things worse (0.775 → 0.730), so choose the calibration per question type.

This is a 0.5B model: the numbers show the pipeline works end to end on a real model and that calibration helps. They are not a ceiling.

### Differences from Gemini Decisions

Text input only for now; explicit `options` / `levels` fields instead of `criteria`; probability quality depends on the model you choose and on calibration — no claim of parity with `gemini-decision-1`.

### Development

`pip install -e ".[dev]" && pytest -q` — fully offline: tests build a tiny tokenizer and a random model on the fly.

License: Apache-2.0.
