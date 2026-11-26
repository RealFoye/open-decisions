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

- **共享前缀 KV 复用**：输入只 prefill 一次，所有问题在同一份 KV cache 上批量打分；同一输入的后续请求直接命中前缀缓存。结果与逐题完整计算数值一致（测试里误差 < 1e-4）。
- **多 token 选项**：选项标签不止一个 token 时（如 `10`、`refund_request`），用 teacher forcing 算整段序列的对数概率，只对这些选项额外跑一轮。
- **概率校准**：温度缩放（所有题型）、Platt 缩放（是非题，顺便修正“爱说 Yes”的偏差）、无标注的 contextual calibration；附带 `opendecisions calibrate` 命令，输出校准前后的 ECE / NLL。
- **多后端 + 编排**：本地 Hugging Face 模型、任意 OpenAI 兼容接口（OpenAI、vLLM、SGLang、Ollama…）、Claude；可以串成 **cascade**（小模型先答，没把握的再升级给大模型）或 **ensemble**（多模型概率平均）。
- **Decisions 风格的接口**：`POST /v1beta/decisions`，请求是一个 `input` 加一组 `questions`，返回带 `confidence` 的类型化答案，和 Gemini Decisions 的形状接近，迁移成本低。

## 快速开始

```bash
git clone <this repo> && cd open-decisions
pip install -e ".[local,server]"        # CPU 用户可先装 CPU 版 torch

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

| 后端 | 概率从哪来 | 共享前缀怎么复用 | 说明 |
|---|---|---|---|
| `local`（Hugging Face transformers） | 真实 token 对数概率，**精确** | 进程内 KV cache：prefill 一次，按批广播；跨请求 LRU 前缀缓存 | 支持 chat template 和纯文本两种 prompt |
| `openai`（任意 OpenAI 兼容接口） | `logprobs` + `top_logprobs`（取前 20）。模型不返回 logprobs 时自动退回到一次请求采样 `n` 个答案，标记为 `estimated` | 共享部分放在消息最前面，命中 OpenAI 自动提示缓存或 vLLM / SGLang 的 prefix caching | 生产环境推荐：`vllm serve <model> --enable-prefix-caching`，再用 `examples/vllm.json` 指过去 |
| `anthropic`（Claude，官方 SDK） | Claude 的 Messages API **不返回 logprobs**，只能估计：`sampling`（结构化输出把答案限定在选项里，采样 k 次，Dirichlet 平滑频率）或 `verbalized`（一次调用让模型给出每个选项的概率） | 输入作为单独的内容块并加 `cache_control`，第一题之后从提示缓存读 | 答案一律 `estimated=true`；verbalized 概率建议先校准再用 |

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

`python benchmarks/bench_prefix.py`。离线、无需下载：用随机权重的 Llama 结构测速度（速度和权重无关）。4 核 CPU，输入 5,753 token，题型按是非 / 四选一 / 五级打分轮换：

| 题数 questions | 逐题调用 naive | 共享前缀（冷） shared, cold | 共享前缀（热，命中前缀缓存） shared, warm | 加速（冷） | 实际计算的 token 数 | 最大 \|Δ log p\| |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 20.70 s | 3.03 s | 1.48 s | 6.8× | 76,387 → 6,679 | 1.1e-05 |
| 50 | 109.09 s | 8.31 s | 6.83 s | 13.1× | 393,777 → 10,383 | 2.3e-05 |
| 100 | 212.55 s | 16.35 s | 14.58 s | 13.0× | 781,687 → 14,899 | 2.3e-05 |

（19M 参数随机权重 Llama，torch 4 线程，batch size 16。最后一列是两种算法的数值差，证明结果一致。）

“逐题调用”就是把每道题当成一次独立的完整请求，这正是只包装 API 的做法在自托管模型上的成本。共享前缀之后，每道题只剩问题本身的 token 要算；剩下的开销主要是 transformers 里把前缀 KV 拼进每个批次的内存拷贝，换成 vLLM / SGLang 这类分页注意力引擎（`openai` 后端 + prefix caching）可以进一步省掉。换真实模型：`--model Qwen/Qwen2.5-0.5B-Instruct`。

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
- **Backends and orchestration.** Local Hugging Face models, any OpenAI-compatible endpoint (OpenAI, vLLM, SGLang, Ollama, …) and Claude, composable as a **cascade** (a small model answers everything; low-confidence answers escalate to the next stage) or an **ensemble** (weighted average of distributions).
- **Decisions-style API.** `POST /v1beta/decisions` (and `/v1/decisions`) with an `input` and a `questions` map; typed answers with `confidence`, `method`, `estimated`, `backend` and `escalated`; `usage` shows how many prefix tokens were reused.

### Quick start

```bash
pip install -e ".[local,server]"
opendecisions serve --model Qwen/Qwen2.5-1.5B-Instruct
curl -s localhost:8000/v1beta/decisions -H 'content-type: application/json' -d @examples/request.json
```

### Where the probabilities come from

| Backend | Probabilities | Prefix reuse |
|---|---|---|
| `local` | exact token log-probabilities | in-process KV cache, prefilled once, LRU across requests |
| `openai` | `logprobs`/`top_logprobs`; if the model returns none, falls back to sampling `n` answers in one request (`estimated: true`) | shared part first, so OpenAI automatic caching or vLLM/SGLang prefix caching kicks in |
| `anthropic` | the Messages API exposes no logprobs, so estimates only: `sampling` (answers constrained to the labels with structured outputs, smoothed frequencies) or `verbalized` (the model states a probability per label) — always `estimated: true` | the input is its own content block with `cache_control` |

Estimated probabilities are labelled as such rather than passed off as exact.

### Benchmark

`python benchmarks/bench_prefix.py` runs offline on a random-weight Llama (timings do not depend on weights): 4-core CPU, 5,753-token input: 100 questions take 212.6 s naively and 16.4 s with a shared prefix (13×), computing 781,687 vs 14,899 tokens, with identical results (max |Δ log p| 2.3e-05). See the table above; the remaining cost after sharing is mostly KV copying inside transformers, which a paged-attention server (vLLM/SGLang via the `openai` backend) avoids.

### Differences from Gemini Decisions

Text input only for now; explicit `options` / `levels` fields instead of `criteria`; probability quality depends on the model you choose and on calibration — no claim of parity with `gemini-decision-1`.

### Development

`pip install -e ".[dev]" && pytest -q` — fully offline: tests build a tiny tokenizer and a random model on the fly.

License: Apache-2.0.
