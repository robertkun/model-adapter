# Bonsai 2 27B：服务端与本机测试

测试日期：2026-10-27。语义判断只在本机做过。服务端测的是命令行生成速度。

## 模型说明

Ternary Bonsai 2 27B 是 PrismML 的三值量化模型，基座是 **Qwen3.8-27B**。网络结构没有改：混合注意力（大约 75% 线性注意力、25% 全注意力）、SwiGLU、RoPE、RMSNorm。它仍然是这个 Qwen 模型，变的是权重的存法。

Bonsai 做的调整是把语言模型权重收成三值 `{−1, 0, +1}`，每 128 个权重配一个 FP16 缩放。发布包大约 1.75 bit/权重。总参数 27.36B，其中语言主干 24.35B，词嵌入和输出头 2.54B，视觉塔 0.46B。官方在思考模式的评测里，平均分大约保住 FP16 基座的 98%。磁盘上的权重大约是全精度的九分之一。

能力范围：文本和图片进、文本出，上下文最长 262144 token，支持工具调用，默认先思考再回答。许可是 Apache 2.0。

适合的场景：

- 要在 CPU 或小显存上跑一个 27B 对话模型。权重只有大约 6–8GB，128GB 内存的服务器和 24GB 统一内存的 Mac 都放得下。
- 需要长上下文、工具调用或看图。这次部署没有接图片。
- 用它做「两句话是不是同一个意思」这种生成式判断。下面的 92.9% 就是这种用法。

它不是专用的向量模型，也不是专用精排模型。大批量检索仍然应该走 embedding 和 rerank。Bonsai 适合需要读懂句子再做决定的场合，单条要比向量检索慢。

两种打包：

| 机器 | 文件 | 原因 |
|---|---|---|
| 服务端 | `PQ2_0`，7.21GB | CentOS 7 上编译了 PrismML 的 llama.cpp，才能跑这种打包 |
| 本机 | `Q2_0`，7.09 GiB | 本机是上游 llama.cpp build 11669，只能跑 `Q2_0` |

两边是同一套权重，打包方式不同。`PQ2_0` 的读入通常更快一些。下面的速度差主要来自硬件，不是来自这个差别。

## 1. 机器配置

### 服务端

Xeon E5-2620 v4，2.10 GHz，Broadwell。2 路 × 8 核 × 2 线程，共 32 个逻辑核，有 AVX2 和 FMA，没有 AVX-512，没有显卡。内存大约 128GB。权重绑在 NUMA node0 的 8 个物理核上。

实际参数：`-ngl 0 -t 8 -c 32768`，采样按思考模式（temperature 1.0，top_p 0.95，top_k 20，min_p 0.05），reasoning effort 为 medium。

### 本机

Apple M4 Pro。14 个 CPU 核（10 性能核 + 4 能效核），20 个 GPU 核，24GB 统一内存，Metal。llama.cpp build 11669。

测速时全部层放在 Metal 上（`-ngl 99`），10 线程，上下文 4096，Flash Attention 开着。接小龙虾时上下文改为 32768，因为工具提示本身大约 1.7 万 token，4096 装不下。

语义判断另外关掉了思考，temperature 为 0，只让模型输出 `hit` 或 `miss`。句子 A 或句子 B 为空时不调用模型，直接记为 miss。

## 2. 每秒多少 token

| | 读入 | 生成 |
|---|---|---|
| 服务端，对话实测 | 4.8 t/s，另一次 5.3 t/s | 2.4 t/s，另一次 2.5 t/s |
| 本机，llama-bench（512 token 读入 / 128 token 生成） | 121.4 t/s | 21.5 t/s |

本机生成大约是服务端的 9 倍，读入大约是 24 倍。服务端这个速度只适合偶尔问一句。本机短回答可以交互；一旦打开默认的长时间思考，生成时间会明显变长。

## 3. 语义判断的准确率

本机 BONSAI 标签，同一套句对。同义为 hit，不同义为 miss，再和人工标注比较。

全部用例（368 条，其中期望 hit 82 条）

- 模型判断 hit：82/368 = 22.3%
- 与标注一致：342/368 = 92.9%
- 期望 hit 子集：一致 69/82 = 84.1%
- 期望 miss 子集：一致 273/286 = 95.5%

模型判出的 hit 条数和期望 hit 一样，都是 82 条，但不是同一批句子。真正同义的 82 条里判对 69 条，漏掉 13 条。不同义的 286 条里判对 273 条，误判成同义 13 条。所以 hit 的精确率和召回率都是 69/82 = 84.1%。它更不容易把不同义的句子说成同义。

服务端没有跑这 368 条。

## 4. 本机启动

命令行对话用 `./deploy/bonsai2-mac/chat.sh`。小龙虾不走这个进程，它要一个 OpenAI 兼容的 HTTP 服务。本机 Homebrew 自带的 Node 是 23，小龙虾要求 Node 24.22.1+ 或 26.12.1+，所以 Node 26.12.1 装在 `~/.local/node-v26`，`openclaw` 命令在 `/opt/homebrew/bin/openclaw`。

24GB 内存同时只跑一份权重。开服务之前先退出 `chat.sh`。

### 启动模型服务

在 `model-adapter` 目录执行：

```bash
llama-server \
  -m deploy/bonsai2-mac/models/Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf \
  --alias bonsai-2-27b \
  --host 127.0.0.1 --port 8091 \
  -ngl 99 -t 10 -c 32768 -np 1 -fa on \
  --reasoning off \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0
```

看到 `listening on http://127.0.0.1:8091` 后再做下一步。模型名是 `bonsai-2-27b`。

### 安装并配置小龙虾

第一次没有 `openclaw` 命令时安装。已安装则跳过。

```bash
# Node 26.12.1 解压到 ~/.local/node-v26 之后
export PATH="$HOME/.local/node-v26/bin:$PATH"
npm install -g --prefix "$HOME/.local/node-v26" openclaw@latest --allow-scripts=openclaw
```

把本机模型写成默认模型。`apiKey` 填任意非空字符串，本机服务不校验。

```bash
openclaw onboard --non-interactive --accept-risk --skip-health \
  --auth-choice custom-api-key \
  --custom-base-url "http://127.0.0.1:8091/v1" \
  --custom-api-key "local" \
  --custom-provider-id local \
  --custom-compatibility openai \
  --custom-model-id "bonsai-2-27b"
```

配置写在 `~/.openclaw/openclaw.json`。这次实际用的值：

- 默认模型：`local/bonsai-2-27b`
- 地址：`http://127.0.0.1:8091/v1`
- 接口：`openai-completions`
- 上下文：32768
- 单次最多生成：2048 token
- 超时：300 秒
- 思考：关闭

地址必须带 `/v1`。上下文要和 `llama-server` 的 `-c 32768` 一致。

### 打开小龙虾

```bash
openclaw gateway
```

另一个终端执行 `openclaw dashboard`，浏览器打开控制页后再发消息。小龙虾的工具提示大约 1.5 万 token，第一遍读入要两分钟左右，生成大约 18 token/秒。页面会一直显示「等待响应」，直到这一轮工具调用结束。

本机没有生图模型。`image_generate` 只是工具名，画不出图片。

### 停止

在 `llama-server` 那个终端按 Ctrl+C。然后：

```bash
openclaw gateway stop --force
```

## 结论

Bonsai 2 27B 是 Qwen3.8-27B 的三值版本，结构不变，体积大约降到九分之一，评测成绩大约保住基座的 98%。拿它做句对同义判断，本机 368 条里和标注一致的有 92.9%。

速度取决于设备。同一模型在 E5-2620 v4 上生成大约 2.4 t/s，在 M4 Pro 的 Metal 上生成 21.5 t/s。服务端适合把模型放得下、偶尔调用。要看生成速度，用本机。要做大批量语义召回，仍然用向量和精排，不要用这个生成模型逐条判断。
