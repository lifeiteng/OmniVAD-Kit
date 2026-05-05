# OmniVAD

[![PyPI](https://img.shields.io/pypi/v/omnivad)](https://pypi.org/project/omnivad/)
[![npm](https://img.shields.io/npm/v/omnivad)](https://www.npmjs.com/package/omnivad)
[![License](https://img.shields.io/github/license/lifeiteng/OmniVAD-Kit)](LICENSE)

[English](README.md) | **中文**

基于 [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) 的跨平台语音活动检测与音频事件检测工具包。

**三个模型，一套工具，全平台运行：**

| 模型 | 功能 | 输出 |
|------|------|------|
| **VAD** | 语音检测（非流式） | 语音时间戳 |
| **Stream-VAD** | 实时语音检测（逐帧） | 每帧语音概率 |
| **AED** | 音频事件检测（非流式） | 语音 / 歌声 / 音乐 时间戳 |

所有模型基于 DFSMN 架构，每个约 2.2MB（~588K 参数），支持 100+ 种语言。

## 安装包

### Python (`omnivad/`)

PyPI 包，内含原生 C 绑定（ncnn 后端），模型已打包在 wheel 中。

```bash
pip install omnivad
```

**命令行：**

```bash
omnivad audio.wav                        # VAD + AED → audio.TextGrid
omnivad audio.wav -o out.json            # 输出 JSON
omnivad audio.wav -o out.srt             # 输出 SRT 字幕
omnivad audio.wav -o out.vtt             # 输出 WebVTT 字幕
omnivad audio.wav -f srt                 # 指定格式 (textgrid/json/srt/vtt)
omnivad audio.wav -m vad                 # 仅 VAD
omnivad audio.wav -m aed                 # 仅 AED（语音/歌声/音乐）
omnivad long.wav --chunk 600 --overlap 2 # 长音频分块处理
python -m omnivad audio.wav              # 同样可用
```

**Python API：**

```python
from omnivad import OmniVAD, OmniStreamVAD, OmniAED
import numpy as np

vad = OmniVAD()

# 文件路径 — 自动加载为 float32 [-1,1]
result = vad.detect("audio.wav")
# {'duration': 2.24, 'timestamps': [(0.26, 1.82)]}

# Float32 数组 [-1.0, 1.0] — 来自 soundfile、torchaudio、librosa
result = vad.detect(float32_array)

# Int16 数组 — 来自原始 WAV、麦克风 PCM
result = vad.detect(np.array([...], dtype=np.int16))

# 长音频 — 分块处理，带重叠
# overlap_seconds 必须小于 chunk_seconds
result = vad.detect("long.wav", chunk_seconds=600, overlap_seconds=2)

# 流式 VAD — 实时处理，每次输入 160 个 int16 采样（10ms）
svad = OmniStreamVAD()
frame = None
while frame is None:
    frame = svad.process(pcm_160_int16)
# StreamResult(time=0.420s, confidence=0.95, is_speech=True)

# FastClone — 共享模型权重，每个流只需极少内存
clone = svad.clone()  # 瞬时创建，~0 内存开销
clone.process(pcm_160_int16)  # 完全独立的状态

# AED — 语音 + 歌声 + 音乐
aed = OmniAED()
events = aed.detect("audio.wav")
# {'duration': 22.0, 'events': {'speech': [...], 'singing': [...], 'music': [...]}}
```

**支持平台：** macOS (arm64/x86_64)、Linux (x86_64/aarch64)、Windows (x86_64)

### C/C++ 原生库 (`native/`)

统一 C API，[ncnn](https://github.com/Tencent/ncnn) 后端，单头文件，单库文件。

```c
#include "omnivad.h"

int err = OMNI_OK;

// VAD — 完整音频 → 语音片段
OmniVadHandle vad = omni_vad_create("vad.omnivad", &err);
omni_vad_detect_int16(vad, pcm, num_samples, &config, &segments, &count);
// segments[0] = { start: 0.44, end: 1.82 }

// 流式 VAD — 实时处理，每帧 10ms
OmniStreamVadHandle svad = omni_stream_vad_create("stream-vad.omnivad", 0.5f, &err);
omni_stream_vad_process(svad, pcm_160_samples, 160, &result);
// result.confidence = 0.95, result.is_speech = true

// FastClone — 跨流共享模型权重
OmniStreamVadHandle clone = omni_stream_vad_clone(svad, &err);
omni_stream_vad_process(clone, other_pcm, 160, &result);  // 独立状态

// AED — 语音 + 歌声 + 音乐检测
OmniAedHandle aed = omni_aed_create("aed.omnivad", &err);
omni_aed_detect_int16(aed, pcm, num_samples, &config, &segments, &count);
// segments[0] = { start: 0.09, end: 12.32, cls: OMNI_AED_MUSIC }
```

**编译：**

```bash
# 前置依赖：cmake、ncnn（brew install ncnn）
cd native
cmake -B build && cmake --build build -j$(nproc)

# 测试
./build/test_all ../models/ audio.wav
```

**支持平台：** macOS (arm64/x86_64)、Linux (x86_64/aarch64)、Windows (x86_64)、Android (armeabi-v7a/arm64-v8a)

### TypeScript/JavaScript (`packages/omnivad/`)

同时支持**浏览器**和 **Node.js**，基于 ncnn WebAssembly。**零依赖**，模型已打包。

```ts
import { OmniVAD, OmniStreamVAD, OmniAED } from 'omnivad';

// 非流式 VAD — 模型从内置 WASM 自动加载
const vad = await OmniVAD.create();
const result = vad.detect(audioFloat32Array);  // Float32Array [-1.0, 1.0]
// { duration: 2.32, timestamps: [[0.44, 1.82]] }

// 也接受 Int16Array（原始 PCM）
const result2 = vad.detect(pcmInt16Array);

// 流式 VAD — 逐帧处理或全音频批量模式
const svad = await OmniStreamVAD.create();
const frame = svad.processFrame(pcm160);  // 缓冲足够前返回 null
const full = svad.detectFull(audioFloat32Array);
// { probabilities: Float32Array(...), numFrames: 98, duration: 1.0 }

// AED — 语音 + 歌声 + 音乐
const aed = await OmniAED.create();
const events = aed.detect(audioFloat32Array);
// { duration: 22.0, events: { speech: [...], singing: [...], music: [...] }, ratios: { ... } }
```

**编译：**

```bash
cd packages/omnivad
pnpm install && pnpm build
# 输出：dist/index.js + dist/index.cjs + dist/index.d.ts + dist/wasm/*
```

## 线程安全

| 组件 | 共享 handle | 独立 handle | 说明 |
|------|:---:|:---:|------|
| **OmniVAD** | **安全** | **安全** | `ncnn::Net` 只读；每次调用创建独立的 `Fbank` 和 `Extractor` |
| **OmniAED** | **安全** | **安全** | 与 VAD 相同的架构 |
| **OmniStreamVAD** | **不安全** | **安全** | 内部可变状态（`audio_buffer`、`cache`、`frame_offset`） |

**使用指南：**

- `OmniVAD` 和 `OmniAED` 实例可以安全地在多线程间共享进行并发推理。Python 的 `detect(..., workers=N)` 参数已使用此模式。
- `OmniStreamVAD` 实例**不可**跨线程共享。并行流式处理时需每个线程创建独立实例。
- Handle 创建（`omni_*_create`）应顺序执行 — ncnn 的模型加载不适用于高并发初始化。
- 不要在其他线程使用 handle 时调用 `close()` / `destroy()`。

**运行线程安全测试：**

```bash
# Python
pytest tests/test_thread_safety.py -v

# C++（需要 ncnn）
./native/build/test_thread_safety models/ tests/data/hello_en.wav [threads] [repeats]
```

## 音频输入

高级 API 仅接受 16kHz 单声道音频。

- Python 和 TypeScript 的 `OmniVAD` / `OmniAED` 接受归一化 `float32`/`Float32Array`（范围 `[-1, 1]`）和 `int16` / `Int16Array`。
- Python 的 `OmniStreamVAD.process()` 接受 `int16` 块，内部也会转换归一化 `float32` 块。
- TypeScript 的 `OmniStreamVAD.processFrame()` 期望 `Int16Array` 块。
- `OmniStreamVAD.detect_full()` / `detectFull()` 接受完整音频缓冲区，内部处理归一化。
- C API 比 Python/TypeScript 封装更底层。具体输入约定请参见 [`native/include/omnivad.h`](native/include/omnivad.h)。

## 音频处理流水线

```
16kHz PCM → Fbank (80维, 25ms窗, 10ms步长) → CMVN → DFSMN → Sigmoid → 后处理 → 片段
                    Povey 窗                      μ/σ    ~2.2MB   [0,1]   4状态机
                    预加重 0.97                                          合并/拆分/扩展
```

## 流式 VAD — `OmniStreamVAD`

长音频场景（直播流、几小时录音、实时字幕），`OmniStreamVAD` 逐帧处理音频并在
**确认段边界的当帧**直接输出段事件 —— 与上游 [FireRedVAD `FireRedStreamVad`](https://github.com/FireRedTeam/FireRedVAD/blob/main/fireredvad/stream_vad.py) bit-identical 一致。

每次成功的 `process()` 调用同时返回逐帧概率 + 段边界事件：

| 字段 | 含义 |
|------|------|
| `confidence` | 模型原始概率 `[0, 1]` |
| `smoothed_prob` | 在 `smooth_window_size` 帧上做的因果移动平均 |
| `is_speech` | `smoothed_prob >= threshold` |
| `is_speech_start` | 在确认新 SPEECH 段的当帧为 `True` |
| `is_speech_end` | 在确认 SPEECH 段结束的当帧为 `True` |
| `frame_idx` | 1-based 帧索引（× 0.01 即秒）|
| `speech_start_frame` | 段起点（`is_speech_start` 时有效）|
| `speech_end_frame` | 段终点（`is_speech_end` 时有效）|

### 配置（默认值与上游 FireRedVAD 完全一致）

| 参数 | 默认值 | 含义 |
|------|-------|------|
| `threshold` | `0.5` | 语音激活阈值 |
| `smooth_window_size` | `5` | 因果移动平均窗口（帧）|
| `pad_start_frame` | `5` | 确认 START 后向前扩展 N 帧 |
| `min_speech_frame` | `8` | 确认 START 所需最少连续 speech 帧（~80ms）|
| `max_speech_frame` | `2000` | SPEECH 累计超过此数强制切分（~20s）|
| `min_silence_frame` | `20` | 确认 END 所需最少连续 silence 帧（~200ms）|

### Python

```python
from omnivad import OmniStreamVAD
import numpy as np

vad = OmniStreamVAD()                              # 用上游默认值
pcm = np.fromfile("speech.pcm", dtype=np.int16)

for i in range(0, len(pcm), 160):                  # 10ms 帧
    result = vad.process(pcm[i : i + 160])
    if result is None:
        continue
    if result.is_speech_start:
        print(f"START @ {result.speech_start_frame * 0.01:.2f}s")
    if result.is_speech_end:
        print(f"END   @ {result.speech_end_frame * 0.01:.2f}s")

# 或一次拿 [(start_sec, end_sec), ...]:
segments = OmniStreamVAD().detect_segments("speech.wav")
```

### TypeScript

```typescript
import { OmniStreamVAD } from "omnivad";

const vad = await OmniStreamVAD.create();
for (let i = 0; i + 160 <= pcm.length; i += 160) {
    const result = vad.processFrame(pcm.subarray(i, i + 160));
    if (!result) continue;
    if (result.isSpeechStart) {
        console.log(`START @ ${(result.speechStartFrame * 0.01).toFixed(2)}s`);
    }
    if (result.isSpeechEnd) {
        console.log(`END   @ ${(result.speechEndFrame * 0.01).toFixed(2)}s`);
    }
}
```

### 与 `merge_chunks` 配合

`OmniStreamVAD` 输出原始 VAD 段。如果想打包成 Whisper 30s chunk 给下游 ASR，
把段对喂给 `merge_chunks`（见下一节）。

## Chunking（分块） — `merge_chunks` / `mergeChunks`

VAD 输出一组语音 `(start, end)` 片段后，chunking 工具把它们组合成有时长上限的 chunk，
适合下游 ASR / 强制对齐 / TTS 使用。这是一个**纯函数**，不依赖任何模型 — Python 走
`ctypes`、TypeScript 走 Emscripten WASM、C 直接调用。三端共用一份位于
`native/src/chunking.cpp` 的 C 实现。

```python
from omnivad import merge_chunks
chunks = merge_chunks(timestamps, max_chunk_secs=30.0, mode="greedy")
```

```ts
import { mergeChunks } from "omnivad";
const chunks = await mergeChunks(timestamps, { maxChunkSecs: 30.0, mode: "longest_gap" });
```

### 流水线（5 步；Step 1-2 与 Step 4-5 两种 mode 共用）

```
输入（已排序的 segments）
  │
  ├─ Step 1：丢弃 duration < min_speech_secs 的片段
  │
  ├─ Step 2：预合并 gap < min_silence_secs 的相邻片段
  │          （级联合并；重叠时取 max(end)）
  │
  ├─ Step 3：打包成 chunk  ─┬─ mode = "greedy"
  │                          │     顺序追加；当下一段会让 chunk 超过 max_chunk_secs
  │                          │     或 gap > max_gap_secs 时切分
  │                          │
  │                          └─ mode = "longest_gap"
  │                                递归在最长 gap 处切分，直到每个 chunk 的 span ≤ max_chunk_secs
  │
  ├─ Step 4：对仍超过 max_chunk_secs 的 chunk 做等长硬切
  │          （仅当单个 segment 自身就超过 max_chunk_secs 时触发）
  │
  └─ Step 5：应用 pad_onset_secs（clamp 到 ≥ 0）和 pad_offset_secs
             输出 chunks: (start, end, seg_start_idx, seg_count)
```

### 两种 mode 对比

| 属性 | `greedy`（默认） | `longest_gap` |
|---|---|---|
| 策略 | 顺序追加直到下一段溢出 | 在最长内部 gap 处递归切分，直到每个 chunk 满足 `max_chunk_secs` |
| 是否受 `max_chunk_secs` 约束 | **是** —— 硬上限 | **是** —— 递归在 chunk span ≤ `max_chunk_secs` 时停止 |
| 切分位置 | 第一个溢出点 | 超长 span 内最长的停顿处 |
| 是否使用 `max_gap_secs` | **是** —— 首个 `gap > max_gap_secs` 处切分 | **是** —— 递归只在没有任何内部 gap 超过 `max_gap_secs` 时停止 |
| 单 seg > `max_chunk_secs` | Step 4 等长硬切兜底 | 同上 —— Step 4 兜底 |
| 确定性 | 确定 | 确定；并列时取**最左** |
| 推荐用途 | **Whisper / whisperX 风格 ASR**（固定长度输入，需 padding 到 30s） | **接受变长输入的模型** —— 强制对齐、TTS、Encoder 风格 ASR。在自然停顿处切分，无需 padding 到固定长度。 |

同输入两种 mode 对比（`max_chunk_secs=20`）：

```
输入 (max_chunk_secs = 20):
  seg 0 = (0, 5)
  seg 1 = (8, 10)     与 seg 0 的间隔 = 3
  seg 2 = (20, 25)    与 seg 1 的间隔 = 10   ← 更长

greedy
  起始 cur = (0, 5)
  接受 seg 1                   → cur = (0, 10)   [长度 10 ≤ 20 ✓]
  下一段 seg 2 would_exceed:    25 - 0 = 25 > 20  → 切分
  chunks: [(0, 10, 0, 2), (20, 25, 2, 1)]

longest_gap
  span = 25 > 20               → 必须切分
  最长 gap = 10 在索引 1        → 在 seg 1 与 seg 2 之间切
    左半 = [seg 0, seg 1]   span = 10 ≤ 20 ✓ → 保留
    右半 = [seg 2]          span = 5  ≤ 20 ✓ → 保留
  chunks: [(0, 10, 0, 2), (20, 25, 2, 1)]
```

（这个最简例子两种 mode 输出一致。当**最长 gap 不在第一个溢出点**时两者会出现差异。）

### `seg_start_idx` / `seg_count` 语义

这两个字段索引的是 **Step 1+2 之后**的片段视图 —— 被 `min_speech_secs` 丢弃和被
`min_silence_secs` 预合并的段不计入索引空间。两种 mode 都遵循此约定。

### 默认值

`omni_chunk_config_default()`（C） / `default_chunk_config()`（Python） /
`DEFAULT_CHUNK_CONFIG`（TS）返回：

| 字段 | 默认值 | 来源 |
|---|---|---|
| `max_chunk_secs` | `30.0` | 秒；与 Whisper 30s 输入窗口对齐 |
| `max_gap_secs` | `INFINITY` | 禁用 |
| `pad_onset_secs` / `pad_offset_secs` | `0.04` / `0.04` | |
| `min_speech_secs` | `0.0` | 对应 VAD `min_speech_frames` |
| `min_silence_secs` | `0.20` | 对齐 VAD `min_silence_frames=20`（10ms 帧移）|
| `mode` | `OMNI_CHUNK_GREEDY` | 向后兼容 |

> **注意 — Python 便利函数默认值与 C/TS 不一致。** `merge_chunks(...)` 的 Python kwargs
> 把 `pad_onset_secs`、`pad_offset_secs`、`min_silence_secs` 都设为 0（最简调用得到原始输出）。
> 若想匹配上表的默认值，请用 `default_chunk_config()` 返回的值显式传入。
> 详见 `tests/test_chunking.py::test_python_convenience_defaults_differ_from_canonical`。

### Whisper / WhisperX 风格 ASR 流水线

`OmniVAD`（整段批处理）+ `merge_chunks(mode="greedy")` 与 WhisperX 的
`Binarize(max_duration=chunk_size)` + 贪心打包行为 1:1 等价。把语音切片
喂给 Whisper 系列 ASR 模型（固定 30s 输入窗口）时使用此 recipe：

```python
from omnivad import OmniVAD, merge_chunks

vad = OmniVAD()                              # threshold=0.4 默认 —— 对 Whisper 更安全
result = vad.detect("long-audio.wav")        # 整段批处理 VAD

chunks = merge_chunks(
    timestamps=result["timestamps"],
    max_chunk_secs=30.0,                     # Whisper 输入窗口
    mode="greedy",                           # WhisperX 行为
    pad_onset_secs=0.04,
    pad_offset_secs=0.04,
    min_silence_secs=0.20,                   # 对齐 VAD min_silence_frames=20
)
# 每个 chunk：{ start, end, seg_start_idx, seg_count }
# 在 [start, end] 切音频，把切片逐个喂给 Whisper。
```

提示：

- 保持默认 `threshold=0.4`。Whisper 对多余的静音 padding 容忍度高，但对
  词首尾辅音被切非常敏感（提到 0.5 容易吞字并触发幻觉）。
- **不要**在这里用 `mode="longest_gap"` —— 那是为变长输入模型（强制对齐、
  TTS）准备的，不是 WhisperX 行为。
- 对超长音频（>1 小时），给 `vad.detect(...)` 传 `chunk_seconds=600, overlap_seconds=2`
  限制峰值内存。

## 模型文件

Python 包、TypeScript 包和本地示例使用的预构建 `.omnivad` 模型包已包含在仓库的 `models/` 目录中。

仅在需要重新导出 ONNX 或重新生成原生资源时，才需要下载上游 FireRedVAD 检查点。

```bash
# 下载上游 PyTorch 模型 + 导出 ONNX
pip install fireredvad
python -m fireredvad.bin.export_onnx --all

# 或直接下载预导出的 ONNX 模型
# fireredvad_vad.onnx              — 非流式 VAD (2.3MB)
# fireredvad_aed.onnx              — 非流式 AED (2.3MB)
# fireredvad_stream_vad_with_cache.onnx — 流式 VAD (2.2MB)

# C/ncnn 使用：用 pnnx 将 ONNX 转换为 ncnn
pip install pnnx
pnnx fireredvad_vad.onnx "inputshape=[1,100,80]"
```

## 本地开发

本节涵盖从源码构建 OmniVAD，以及在同一台机器上把仓库内构建产物喂给其他项目使用 ——
也就是改 C/C++ 核心、Python 封装或 TS 绑定时常用的开发循环。

### 前置依赖

| 目标 | 依赖 | 说明 |
|------|------|------|
| Python wheel | Python 3.10+、CMake 3.15+、C++14 工具链 | `pip install -e .` 走 scikit-build-core，CMake `FetchContent` **会自动拉 ncnn**。 |
| 独立 C/C++ 库 | CMake 3.15+，**预装 ncnn**（`brew install ncnn` 或自行编译） | `native/CMakeLists.txt` **不会** 自动拉 ncnn —— 如果默认搜索不到，需要 `-DNCNN_ROOT=...`。 |
| TypeScript bundle | Node 18+、[pnpm](https://pnpm.io/) | 只构建 `dist/index.{js,cjs,d.ts}`，**不会** 重建 WASM。 |
| WASM 模块 | [emsdk](https://emscripten.org/docs/getting_started/downloads.html)（任意较新版本）| 仅当改了 C/C++ 代码、需要刷新 `dist/wasm/omnivad.wasm` 时需要。 |

### 构建 Python 包（editable install）

```bash
pip install -e ".[dev]"
```

产物：

- `omnivad/libomnivad.{dylib,so,dll}` —— 运行时 `omnivad/_binding.py` 实际加载的共享库。
- `omnivad/models/*.omnivad` —— 模型文件（CMake `install(...)` 复制进来）。
- 当前环境 `site-packages` 的 editable 入口，链回源码目录。

改了 `native/` 下的 **C/C++ 代码** 之后，重跑 `pip install -e .` 触发 CMake 增量重链
（很快）。纯 Python 修改无需重装。

### 构建 TypeScript 包

```bash
cd packages/omnivad
pnpm install
pnpm build          # tsup → dist/index.{js,cjs,d.ts}
pnpm typecheck      # tsc --noEmit
```

这一步 **不会** 重建 WASM，复用 `dist/wasm/` 已有产物。如果只改了 TS，到此为止。

### 构建 WASM 模块（改了 C/C++ 时）

```bash
EMSDK=/path/to/emsdk packages/omnivad/wasm/build.sh
```

脚本直接把 `omnivad.{js,cjs,wasm}` 写到 `packages/omnivad/dist/wasm/`。如果同时也改了
TS，再跑一次 `pnpm build`。

> `EMSDK` 必须指向 emsdk 根目录（包含 `emsdk_env.sh` 与 `upstream/emscripten/` 的目录）。
> 未设置时脚本会直接报错退出。

### 在其他仓库使用本地构建版本

#### Python — `pip install -e <path>`

```bash
# 在目标项目的 venv 里：
pip install -e /abs/path/to/OmniVAD-Kit          # editable，能持续吃到你的改动
# 或者隔离的 wheel 安装：
pip install /abs/path/to/OmniVAD-Kit             # 重新构建并安装一份 wheel
```

`pip install -e` 是开发循环首选 —— 改 C/C++ 后重跑会原地重链 dylib；纯 Python 改动直接生效。

#### TypeScript —— 三种方案，按场景挑

| 方案 | 命令 | 适用场景 |
|------|------|---------|
| **A. Tarball（最接近 npm 真实安装）** | `cd packages/omnivad && pnpm pack`<br>目标项目：`pnpm add /abs/path/omnivad-0.2.8.tgz` | 验证真实消费者的安装路径，干净无 symlink 怪象。 |
| **B. `file:` 协议** | 目标 `package.json`：`"omnivad": "file:../OmniVAD-Kit/packages/omnivad"` | monorepo 风格的就地消费。重建后跑 `pnpm install` 拉新产物。 |
| **C. 全局 link** | `cd packages/omnivad && pnpm link --global`<br>目标项目：`pnpm link --global omnivad` | 跨多个项目快速迭代。注意 peer/hoist 怪象。 |

三种方案都需要 **测试前先重建**：

```bash
cd packages/omnivad
pnpm build                                       # 只改 TS
EMSDK=/path/to/emsdk wasm/build.sh && pnpm build # 改了 C/C++
```

### 改 C/C++ 后完整重建一行流（备忘）

```bash
# 在仓库根目录：
pip install -e .                                       # Python dylib
EMSDK=/path/to/emsdk packages/omnivad/wasm/build.sh    # WASM（.wasm + glue）
( cd packages/omnivad && pnpm build )                  # TS bundle
```

### 独立 C/C++ 构建（用于 native 测试 / 嵌入）

```bash
cd native
cmake -B build -DNCNN_ROOT=/path/to/ncnn   # 仅在 ncnn 不在默认搜索路径时需要
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
./build/test_all ../models ../tests/data/hello_en.wav
```

这条路径与 Python wheel 构建独立 —— wheel 通过 CMake `FetchContent` 拉一个 pinned 的
ncnn，而 `native/` 期待预装好的 ncnn。

### Lint / Format

```bash
ruff check --fix . && ruff format .                    # Python（line-length 120）
( cd packages/omnivad && pnpm typecheck )              # TypeScript
```

## 测试

```bash
# 运行完整 Python 测试套件
pip install -e ".[dev]"
pytest tests -v

# 工具脚本（非 pytest — 需要外部 FireRedVAD 模型）
python tests/generate_reference.py            # 生成 Python 参考数据
python tests/check_timestamp_accuracy.py      # 严格的 C vs Python 对比
python tests/vad_to_textgrid.py audio.wav     # 音频 → TextGrid + RTF 基准测试
```

**精度对比（C/ncnn vs Python，5 个音频 × 3 个模型）：**

| 模型 | 时间戳差异 | 概率差异 | 状态 |
|------|-----------|---------|------|
| VAD | ≤ 0.020s | ≤ 0.001 | 完全匹配 |
| AED（歌声/音乐） | ≤ 0.010s | ≤ 0.013 | 完全匹配 |
| AED（语音） | ≤ 0.030s | ≤ 0.015 | 匹配（ncnn fp16 在 `event.wav` 上的边界情况） |
| Stream-VAD (detect_full) | ≤ 0.010s | ≤ 0.001 | 完全匹配 |

## 项目结构

```
omnivad/
├── omnivad/                         # Python PyPI 包
│   ├── __init__.py                  #   公共 API：OmniVAD、OmniStreamVAD、OmniAED
│   ├── cli.py                       #   CLI 入口（omnivad 命令）
│   ├── _binding.py                  #   libomnivad 的 ctypes 绑定
│   ├── vad.py                       #   OmniVAD（非流式）
│   ├── stream_vad.py                #   OmniStreamVAD（实时）
│   └── aed.py                       #   OmniAED（3 分类）
├── native/                          # C/C++ 库（ncnn 后端）
│   ├── include/omnivad.h            #   统一 C API 头文件
│   ├── src/omnivad.cpp              #   核心实现
│   ├── frontend/                    #   Fbank/FFT/WAV（来自 FireRedVAD）
│   ├── test/                        #   4 个测试程序
│   └── CMakeLists.txt
├── packages/omnivad/                # TypeScript npm 包
│   ├── src/
│   │   ├── vad.ts                   #   OmniVAD（非流式）
│   │   ├── stream-vad.ts            #   OmniStreamVAD（实时）
│   │   ├── aed.ts                   #   OmniAED（3 分类）
│   │   ├── wasm-binding.ts          #   Emscripten/WASM 绑定
│   │   ├── types.ts                 #   公共 TypeScript 类型
│   │   ├── index.ts                 #   包导出
│   │   └── wasm.d.ts                #   WASM 模块声明
│   ├── package.json
│   └── tsconfig.json
└── tests/                           # 测试套件
    ├── test_c_vs_python.py          #   精度：omnivad vs Python 参考
    ├── test_determinism.py          #   重复运行确定性
    ├── test_edge_cases.py           #   边界情况：极短/空/静音输入
    ├── smoke_test.py                #   CI 冒烟测试（导入 + 检测）
    ├── test_memory.sh               #   原生库内存/泄漏检查
    ├── check_timestamp_accuracy.py  #   严格 C vs Python 对比（手动）
    ├── check_native.py              #   原生 C 二进制验证（手动）
    ├── generate_reference.py        #   生成 Python 参考数据
    ├── vad_to_textgrid.py           #   音频 → TextGrid + RTF 基准测试
    └── data/                        #   5 个测试音频 + 参考 JSON
```

## 性能

RTF（实时因子），在 Apple M 系列芯片上测试，越低越快：

| 模型 | RTF | 速度 |
|------|-----|------|
| VAD | ~0.003 | ~330 倍实时 |
| Stream-VAD | ~0.002 | ~500 倍实时 |
| AED | ~0.002 | ~500 倍实时 |

## 来源与致谢

OmniVAD 是基于 [**FireRedVAD**](https://github.com/FireRedTeam/FireRedVAD) 构建的跨平台部署工具包，FireRedVAD 由[小红书](https://www.xiaohongshu.com/)开发。FireRedVAD 提供高质量的语音活动检测模型和轻量级音频事件检测模型，能够区分语音、歌声和音乐。

**原始论文：** [FireRedVAD (arXiv:2603.10420)](https://arxiv.org/abs/2603.10420)

**FireRedVAD 提供：** DFSMN 架构模型（每个约 2.2MB）、Python 推理代码、PyTorch 训练、优秀的 VAD 基准测试结果（FLEURS-VAD-102 F1: 97.57%）。

**OmniVAD 新增：** 统一 C API（ncnn 后端）用于原生部署、TypeScript/JavaScript npm 包（ncnn WebAssembly）用于浏览器和 Node.js、跨平台构建系统、包含精度验证的完整测试套件。

## 许可证

Apache-2.0 — 与上游 FireRedVAD 一致。

## 致谢

- [**FireRedVAD**](https://github.com/FireRedTeam/FireRedVAD) — Kaituo Xu, Wenpeng Li, Kai Huang, Kun Liu（小红书）
- [ncnn](https://github.com/Tencent/ncnn) — 腾讯
- [Emscripten](https://emscripten.org/) — WebAssembly 工具链
