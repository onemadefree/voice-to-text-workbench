# 跨平台 ASR 方案（调研报告）

> 分支：`research/asr-cross-platform`　基线：`v1.0.0`
> 调研日期：2026-10-29　调研方式：三条腿并行（N 卡 / A 卡 / CPU+Windows 打包），
> 关键结论逐条落到官方页面（PyPI JSON API、GitHub Releases API、ROCm/DirectML/ORT/
> CUDA 官方文档、模型卡）核实；**未经官方页面确认的一律标注「未核实」**。
> 代码侧改造设计见另一篇《跨平台 ASR 改造计划》。

---

## 结论先行

### 一、最重要的发现：**模型可以保住，不用换**

| 运行时 | 平台 | Qwen3-ASR | 强制对齐器 | 热词 | 状态 |
|---|---|---|---|---|---|
| **MLX** 0.32.4 | macOS / **Linux** / **Windows** | ✅ 现有 `mlx-audio` 路径可原样复用 | ✅ 同模型 | ✅ | **轮子已存在（已独立核实 PyPI），但 mlx-audio 的算子覆盖需实测** |
| **vLLM** 0.32.0 | Linux（**无 Windows wheel**） | ✅ 原生 `qwen3_asr.py`，**官方 day-0 支持** | ✅ 原生 `qwen3_asr_forced_aligner.py` | ✅ 上下文偏置 | ✅ 官方 recipe + 官方 Docker `qwenllm/qwen3-asr` |
| **transformers** ≥5.13.0 | Linux / Windows | ✅ 官方原生（`Qwen/Qwen3-ASR-0.6B-hf`） | ✅ `AutoModelForTokenClassification` | ✅ | ✅ |
| **sherpa-onnx** 2.0.1 | Linux / Windows / macOS | ✅ 官方提供 Qwen3-ASR-0.6B int8 ONNX | ❌ `timestamps` 为空 | ❌ | ⚠️ 识别保住，**对齐丢了** |

> **MLX 的跨平台轮子（2026-10-29 独立核实 PyPI）：**
> `mlx` 0.32.4 → win_amd64 / win_arm64 / manylinux_2_35(x86_64,aarch64) / macosx arm64；
> `mlx-cpu` → win_amd64 / win_arm64 / manylinux_2_35；
> `mlx-cuda-12` → **win_amd64** + manylinux_2_35；
> `mlx-metal` → 仅 macOS。`mlx-audio` 0.6.3 是纯 Python，任意平台可用。
>
> ⚠️ **安装坑**：`mlx` 的 extras 全部标了 `platform_system == "Linux"`，
> Windows 上 `pip install mlx[cuda]` **不会**装上后端包，必须显式
> `pip install mlx==0.32.4 mlx-cuda-12==0.32.4`，否则只装到 0.6MB 的空壳 core。

### 二、三个硬件平台的现实结论

| 平台 | 首选 | 兜底 | 词级时间戳 |
|---|---|---|---|
| **macOS Apple Silicon** | 现状不变（MLX Metal） | — | ✅ 独立强制对齐器 |
| **Linux + N 卡** | MLX CUDA（`mlx[cuda12]`，需实测）→ vLLM 原生 Qwen3-ASR | sherpa-onnx CUDA | ✅ 保住 |
| **Windows + N 卡** | MLX CUDA（显式装 `mlx-cuda-12`）→ WSL2 跑 vLLM | sherpa-onnx 官方 Windows CUDA wheel | ✅ 保住 |
| **A 卡（Linux/Windows）** | **FunASR 官方 Vulkan 运行时**（Windows/Linux 都有官方 zip） | PyTorch ROCm 官方 wheel / whisper.cpp HIP | ⚠️ **必然降级**（独立强制对齐 → ASR token 时间戳） |
| **纯 CPU** | **sherpa-onnx + Qwen3-ASR-0.6B int8**（RTF 0.077~0.103） | FunASR llama.cpp 单二进制（零 Python） | ⚠️ Qwen3-ASR ONNX 版为空；Paraformer-zh 自带 |

### 三、明确点名的「死路」

1. **`faster-whisper` / CTranslate2 在 AMD 上用 GPU —— 死路。**
   CTranslate2 官方硬件支持页的 GPU 一栏只有 "NVIDIA GPUs with Compute Capability ≥ 3.5"；
   faster-whisper README 写的是 "GPU execution requires **NVIDIA** libraries: cuBLAS for CUDA 12, cuDNN 9"。
   **不要在任何文档里写「faster-whisper + ROCm」这种组合。**
2. **WhisperX 的 GPU 路径 —— 连带死路**（内部就是 faster-whisper）。要词级对齐质量就只能全 CPU。
3. **ONNX Runtime DirectML 当「词级时间戳方案」—— 不满足硬约束。**
   Whisper 的词级时间戳依赖 cross-attention 权重（DTW 对齐），通用 whisper→ONNX 导出默认不输出
   cross-attention；且 DirectML EP 已被官方标记为 "sustained engineering"、禁并行执行。
4. **PyPI 包 `qwen-asr` 0.0.6 做生产部署 —— 别装。**
   它把 `transformers` 钉死在 **4.57.6**、`vllm` 钉死在 **0.14.0**；
   而 vLLM 0.32.0 要求 `transformers<5.20.0,>=5.16.1` —— 直接冲突。
   直接用 `transformers>=5.13.0` + `-hf` 权重，或直接 `vllm serve`。
5. **在 A 卡上找 MLX 的平替 —— 不存在**，MLX 的 CUDA/Windows 后端都是 NVIDIA 专属。

---

## 三、方案总表（关键维度）

| 方案 | 版本/许可 | Windows 原生 | 词级时间戳 | 热词 | 中文准确度 | RTF 量级 |
|---|---|---|---|---|---|---|
| **MLX + mlx-audio**（现状） | 0.32.4 / 0.6.3 / Apache-2.0 | ✅（需实测） | ✅ 独立对齐器 | ✅ | ✅✅ WenetSpeech net/meeting **4.97 / 5.88**（1.7B） | 本机 Metal 已跑通 |
| **vLLM 原生 Qwen3-ASR** | 0.32.0 / Apache-2.0 | ❌ 仅 Linux（用 WSL2） | ✅ 官方对齐器 | ✅ ≤10k tokens | ✅✅ 同上 | 未核实（官方只给并发 128 时 2000× 吞吐） |
| **transformers Qwen3-ASR** | ≥5.13.0 / Apache-2.0 | ✅ | ✅ 官方对齐器 | ✅ | ✅✅ 同上 | 未核实 |
| **sherpa-onnx + Qwen3-ASR int8** | 2.0.1 / Apache-2.0 | ✅ 官方 CUDA wheel | ❌ `timestamps` 空 | ❌ | ✅✅ 同模型 | **RTF 0.077~0.103**（2 线程，官方） |
| **sherpa-onnx + Paraformer-zh int8** | 2.0.1 / Apache-2.0 | ✅ | ✅ **自带 timestamps** | ✅ **原生 hotword** | ✅✅ 中文专用 220M | **RTF 0.073**（1 线程，官方） |
| **FunASR llama.cpp/GGUF** | 1.4.17 / MIT | ✅ 官方 `windows-x64-cuda/vulkan/avx2.zip` | ❌（靠 VAD 给句级） | ✅ | ✅✅ 官方称中文 CER ≈ whisper.cpp 的 **1/3** | 快，官方有 RTFx 340（vLLM batch） |
| **faster-whisper** | 1.2.1 / MIT | ✅ | ✅ `word_timestamps=True` | ❌ **无热词 API** | ⚠️ WenetSpeech **9.86 / 19.11** | CPU: RTF 0.131（small int8, 8线程）/ 0.065（batch8）；GPU: 13min→1m03s（RTX 3070Ti） |
| **whisper.cpp** | v1.9.5 / MIT | ✅ CUDA 预编译包 | ⚠️ `-ml 1`（README 标 experimental） | ❌ | ⚠️ 同 Whisper 家族 | CPU RTF 0.160（small fp32） |
| **PyTorch ROCm + openai/whisper** | torch 2.9.1+rocm7.2.1 | ✅ 官方 wheel（py3.12） | ✅ | ⚠️ 弱 prompt 偏置 | ⚠️ 同 Whisper | 慢一个量级以上 |
| **NVIDIA Parakeet TDT 0.6b** | — / Apache-2.0 | — | ✅ 官方标注 | ❌ | ❌ **仅英文** | 未核实 |
| **NVIDIA Canary 1b / Riva** | — | — | ✅ | ✅ Riva 层 | ⚠️ 中文支持存疑 | 未核实 |

---

## 四、N 卡（NVIDIA / CUDA）

### 4.1 首选：vLLM 原生 Qwen3-ASR（Linux）

官方 vLLM 模型注册表里有**三个** Qwen3-ASR 原生实现：
`Qwen3ASRForConditionalGeneration`（`qwen3_asr.py`）、
`Qwen3ASRForcedAlignerForTokenClassification`（`qwen3_asr_forced_aligner.py`）、
`Qwen3ASRRealtimeGeneration`。官方 README 原话：*"vLLM officially provides day-0 model support for Qwen3-ASR"*。

```bash
# 官方 recipe（cu129 nightly）
uv pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
uv pip install "vllm[audio]"

# 官方 Docker
docker run --gpus all -p 8000:80 --shm-size=4gb -it qwenllm/qwen3-asr:latest
```

`vllm serve` 同时支持 OpenAI `chat/completions`（`audio_url`）和 `/v1/audio/transcriptions`。

### 4.2 Windows + N 卡的三个选择

| 选择 | 说明 |
|---|---|
| **WSL2 跑同一套 vLLM** | vLLM 0.32.0 只有 `manylinux_2_28_x86_64/aarch64`，**PyPI 上没有 Windows wheel** → WSL2 是唯一零改造路径 |
| **MLX CUDA（显式装）** | `pip install mlx==0.32.4 mlx-cuda-12==0.32.4 mlx-audio`；现有代码路径原样复用；**需实测 mlx-audio 的算子覆盖** |
| **sherpa-onnx 官方 Windows CUDA wheel** | `sherpa_onnx-2.0.1+cuda12.cudnn9-cp3XX-cp3XX-win_amd64.whl`；但要**放弃 Qwen3-ASR**，换 Paraformer-zh |

### 4.3 Windows + CUDA 的五个坑（均已核实）

1. **cuDNN DLL 必须手动装，pip 装不上（Linux 限定）**：
   faster-whisper README 原文里 `pip install nvidia-cudnn-cu12==9.*` 标着 *"Install with pip (Linux only)"*，
   Windows 官方建议下载 Purfview `whisper-standalone-win` 的 `libs` release 解压后加进 `PATH`。
2. **vLLM 无 Windows wheel** → 决定了「Windows 原生」与「Qwen3-ASR+vLLM」互斥。
3. **flash-attn 无 Windows wheel**（只有 sdist，需 nvcc）→ Windows 原生 PyTorch 用 SDPA/eager。
4. **torch 2.15.0 的 CUDA 依赖声明为 Linux 专属**（`cuda-toolkit[...]`、`nvidia-cudnn-cu13`、`triton` 全带
   `platform_system == "Linux"`）→ Windows 要自己装 CUDA Toolkit 13.x + cuDNN 9 并配 `PATH`。
5. **sherpa-onnx 的 Windows CUDA wheel 是否自带 cuDNN DLL —— 未核实**（wheel tag 只写 `cudnn9`），
   **落地前必须在目标机实测一次 `import sherpa_onnx` + 跑一次 GPU 推理**。

### 4.4 版本对齐（官方数据）

| CUDA Toolkit | Linux 驱动 | Windows 驱动 |
|---|---|---|
| 12.9 GA | ≥575.51.03 | ≥576.02 |
| 12.6 GA | ≥560.28.03 | ≥560.76 |
| 12.4 GA | ≥550.54.15 | ≥551.78 |

- **Minor Version Compatibility**：同大版本内 Linux 驱动 ≥450.80.02 / Windows ≥452.39 即可跑更新的 CUDA。
- **MLX CUDA 后端额外要求**：SM ≥ 7.5、驱动 ≥ **550.54.14**、CUDA ≥ 12.0、**glibc ≥ 2.35**、Python ≥ 3.10。
  ⚠️ **glibc ≥ 2.35 = Ubuntu 22.04+** → CentOS 7 / RHEL 8（glibc 2.28）**装不了 MLX 的任何后端**。
- **CTranslate2**：只支持 CUDA 12 + cuDNN 9；CUDA11+cuDNN8 → 降级 `ctranslate2==3.24.0`；
  CUDA12+cuDNN8 → `4.4.0`。

---

## 五、A 卡（AMD）

### 5.1 结论：能跑，但生态是「能跑、没人管」

AMD 官方的 ROCm 文档几乎只覆盖 LLM / ComfyUI / SD，**没有 ASR 专项支持文档**。
真正的可用组合只剩三条：

| 路线 | 平台 | 现成产物 | 时间戳 | 判定 |
|---|---|---|---|---|
| **FunASR 官方 Vulkan 运行时** | Windows + Linux | ✅ 官方 release：`funasr-llamacpp-windows-x64-vulkan.zip`、`funasr-llamacpp-linux-x64-vulkan.zip`、`…windows-x64-cuda.zip`、`…cuda-blackwell(sm_120).zip`、`…avx2.zip` | ❌ 靠 VAD 给句级 | **最省事的 A 卡 GPU 路线**（比 whisper.cpp 官方镜像还直接） |
| **whisper.cpp HIP** | Linux | 官方 Docker `ghcr.io/ggml-org/whisper.cpp:main-vulkan`；自编 `-DGGML_HIP=1 -DAMDGPU_TARGETS="gfx1201"` | ✅ `-ml 1`（experimental） | README 点名 gfx1100/1101/1201 |
| **PyTorch ROCm** | Linux + Windows | 官方 wheel：ROCm 7.2.1 / torch 2.9.1+rocm7.2.1 / **py3.12** / 驱动 26.2.2 | ✅ | Windows 上不用编译；但慢、要自己写模型加载 |

### 5.2 ROCm 官方支持的消费级显卡（Linux，已核实）

| 显卡 | 架构 | 官方支持 |
|---|---|---|
| RX 9070 XT / 9070 GRE / 9070 | RDNA4 / gfx1201 | ✅ |
| RX 9060 XT / 9060 | RDNA4 / gfx1200 | ✅ |
| RX 7900 XTX / 7900 XT / 7900 GRE | RDNA3 / gfx1100 | ✅ |
| RX 7800 XT / 7700 XT / 7700 | RDNA3 / gfx1101 | ✅ |
| Radeon VII | GCN5.1 / gfx906 | ❌ |

补充：消费级 Radeon 在 Linux 上**只支持** Ubuntu 24.04.4 / 22.04.5 / RHEL 10.1 / 9.7；
未收录的卡可通过 TheRock nightly 社区启用。

### 5.3 Vulkan 的已知质量坑（选型必读）

[whisper.cpp issue #2400「The recognition results with Vulkan are so bad」](https://github.com/ggml-org/whisper.cpp/issues/2400)：
2026-02 有用户报告 **AMD 6800XT + Mesa/NVK** 下出现「随机大小写」等识别退化，2026-09 被 bot 以 stale 关闭。
→ **验收标准里必须包含「输出质量抽检」，不能只看 RTF。**

### 5.4 DirectML 的致命点

官方文档（已核实）：DirectML 1.15.2、支持 ONNX opset ≤ 20、**不支持 memory pattern 优化与并行执行**
（必须 `DisableMemPattern` + `ORT_SEQUENTIAL`）、要求 DX12（AMD GCN 1 代及以上，即 RX 7000/9000 全符合）、
且 **"DirectML is in sustained engineering"**，新特性转向 WinML。

**但它拿不到词级时间戳**（导出不含 cross-attention）→ 不满足本项目硬约束。

### 5.5 A 卡的词级时间戳必然降级

| 来源 | A 卡上是否可用 |
|---|---|
| Qwen3-ForcedAligner（独立强制对齐） | ❌ 只有 MLX |
| whisper.cpp `-ml 1` | ✅（experimental） |
| FunASR timestamp | ✅（但 SenseVoice 的 `timestamps` 为空） |
| ONNX whisper + DirectML | ❌ |

→ **产品口径必须写清楚：A 卡用户拿到的「词级时间戳」是 ASR 自带 token 时间戳，
不是独立强制对齐，边界词切分质量低于 N 卡。**

---

## 六、纯 CPU 路线

### 6.1 官方实测 RTF（注意口径不同，不可直接横比）

| 方案 | 模型 / 量化 | 硬件 / 线程 | 音频 → 耗时 | RTF | 来源 |
|---|---|---|---|---|---|
| sherpa-onnx | **Qwen3-ASR-0.6B int8 ONNX** | CPU（型号未标）/ 2 线程 | 272.4s 中文 → 20.97s | **0.077** | [sherpa-onnx Qwen3-ASR](https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/pretrained.html) |
| sherpa-onnx | Qwen3-ASR-0.6B int8 | CPU / 2 线程 | 334.2s 英文 → 34.48s | 0.103 | 同上 |
| sherpa-onnx | Paraformer-zh int8 | CPU / 1 线程 | 7.8s → 0.573s | **0.073** | [Paraformer](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html) |
| sherpa-onnx | SenseVoice int8 | CPU / 1 线程 | 12.7s → 1.392s | 0.109 | [SenseVoice](https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html) |
| faster-whisper | small int8 | i7-12700K / 8 线程 | 13min → 1m42s | 0.131 | [faster-whisper](https://pypi.org/project/faster-whisper/) |
| faster-whisper | small int8 **batch=8** | 同上 | 13min → 51s | **0.065** | 同上 |
| whisper.cpp | small fp32 | 同上 | 13min → 2m05s | 0.160 | 同上 |
| whisper.cpp + OpenVINO | small fp32 | 同上 | 13min → 1m45s | 0.135 | 同上 |
| openai/whisper | small fp32 | 同上 | 13min → 6m58s | 0.535 | 同上 |

> faster-whisper 官方警告：对比必须同 beam size（它默认 5、openai/whisper 默认 1）与同线程数，
> 否则不可比。上表是官方自建的**同一段 13 分钟音频**横比，是唯一可信的基准。
> sherpa-onnx 页面**未标注 CPU 型号**，只能当量级参考。

**推论（按规模外推，标为推测）**：1 小时音频，faster-whisper int8 small ≈ 8 分钟；
large-v3 int8 大致是 small 的 3~6 倍，即 1 小时 ≈ 25~50 分钟。

### 6.2 Intel AMX

- ggml 里 AMX 路径**存在于代码中**（`ggml-cpu/amx/amx.h`），但需要 Sapphire Rapids 级 CPU
  且要命中 int8 量化路径。
- ONNX Runtime 官方 wheel **没有 oneDNN-DNNL 包** → 想在 ORT 上吃 AMX 需源码编译。
- **AMX 在 ASR 场景的实际加速倍数 —— 未核实**（只拿到 LLM 场景的二手 3–5×，不采信）。
  可落地的表述是：**在本项目可选的栈里只有 OpenVINO 路线吃得到 AMX**。

### 6.3 热词：中石油场景的关键

| 方案 | 热词 |
|---|---|
| **FunASR Paraformer** | ✅ **原生**：`AutoModel(model="paraformer-zh", hotword="关键词 20")` |
| sherpa-onnx | ✅ Zipformer 系支持 `hotwords_file` / modified beam search；SenseVoice 有第三方 hotword 权重（非官方） |
| Whisper / faster-whisper | ❌ 原生无热词 API |
| Qwen3-ASR | ✅ 上下文偏置（≤10k tokens） |

→ 术语/人名（「苏里格」「克拉玛依」这类）在 Whisper 上最容易翻车，
**若 CPU 路线要支持热词，Paraformer-zh 是唯一原生支持的组合。**

### 6.4 最低配兜底：零 Python 的单二进制

```
STT-离线版/
├─ llama-funasr-sensevoice.exe      ← FunASR llama.cpp 发行版
├─ gguf/
│  ├─ sensevoice-small-q8.gguf      ← 中文识别
│  └─ fsmn-vad.gguf                 ← 内置 VAD，负责静音切块
└─ models/                          ← 可选，放 Paraformer-zh 提精度
```

官方 README：*"single self-contained binary on CPU"*、*"Built-in FSMN-VAD, no Python at runtime"*，
Windows 有现成 PowerShell 下载示例。代价是 SenseVoice 的 `timestamps` 为空 → 句级时间靠 VAD。

---

## 七、Windows 打包与分发

### 7.1 打包方式横评

| 方案 | 结论 |
|---|---|
| **PyInstaller onedir** | ✅ **首选**。⚠️ PyInstaller 官方 371 个 hook 里**没有 onnxruntime hook** → 要自己写 `hook-onnxruntime.py`（`collect_dynamic_libs('onnxruntime')`） |
| PyInstaller onefile | ❌ 启动解压到 temp → 杀软误报高、启动慢、**模型不能内置**（GB 级解压不可接受） |
| **内嵌 Python 便携版** | ✅ **与 onedir 并列首选**（改代码频繁时更优，业务代码零改动）。⚠️ embeddable 包不含 pip、需改 `python311._pth` 打开 `import site` |
| **Nuitka** | ✅ 备选（杀软误报率明显低于打包器）。⚠️ Windows 必须装 VS 2022（含英文语言包）；MinGW64 不支持 Python 3.13+ |
| Docker Desktop | ❌ 对「不懂技术的最终用户」是灾难，GPU 还要 WSL2 后端 |
| WSL2 | ⚠️ 门槛高、GPU 转发坑多；但**是 Windows 上跑 vLLM 的唯一路径** |

### 7.2 模型分发：外置，不塞 exe

```
STT/
├─ 启动.bat / 语音转文字.exe
├─ models/
│  ├─ Qwen3-ASR-0.6B-8bit/                    ← MLX 路径
│  ├─ sherpa-onnx-qwen3-asr-0.6B-int8-.../    ← ONNX 路径
│  └─ SenseVoiceSmall-q8.gguf + fsmn-vad.gguf ← 兜底
├─ wheels/      ← 离线 pip 依赖
└─ app/          ← 源码或打包产物
```

- 定位顺序：`exe 同级 models/` → 环境变量 `STT_MODELS_DIR` → `~/.cache`。
  ⚠️ **不能用 `sys._MEIPASS`**（PyInstaller 指向临时解压目录）。
- 现有 `scripts/fetch_models.py` 已有「HF / ModelScope 双源 + 断点续传」，
  扩展成「按平台下载对应格式」即可。

### 7.3 离线安装

```bash
# 在联网机器上为 Windows x64 / CPython 3.11 抓全依赖
pip download --dest wheels/win311 \
  --platform win_amd64 --python-version 3.11 --implementation cp --abi cp311 \
  --only-binary=:all: -r requirements-win.txt

# 目标机离线安装
pip install --no-index --find-links wheels/win311 -r requirements-win.txt
```

⚠️ 用 `--platform` 时**必须**同时给 `--only-binary=:all:` 或 `--no-deps`。
锁文件建议用 `uv pip compile pyproject.toml --extra win-cpu -o requirements-win.lock` 生成，别手写。

### 7.4 中文 Windows 环境的坑（逐条带官方依据）

| 坑 | 事实 | 实操 |
|---|---|---|
| **GBK 编码** | Windows 系统编码仍是 legacy ANSI Code Page（GBK） | 启动时 `PYTHONUTF8=1`；`sys.stdout.reconfigure(encoding="utf-8")`；所有 `open()` 显式 encoding；`subprocess` 传 `encoding="utf-8", errors="replace"` |
| **MAX_PATH 260** | 需管理员开组策略 "Enable Win32 long paths" | **安装目录要短**（`C:\STT`）→ 更加坐实「onedir + 短路径」 |
| **杀软误报** | PyInstaller 文档自曝 onefile 有「Security Validation in onefile Mode」限制 | onedir > Nuitka > onefile；给 exe **代码签名**；提前给 IT/安全部门报备白名单；内部共享盘分发而非邮件附件 |
| **ORT 的 VC++ 运行时** | Windows 版 onnxruntime 必须装 Visual C++ 2019 runtime | 交付包内附 `vcredist.x64.exe`，启动器里静默安装 |
| **CUDA/cuDNN 的 PATH** | Windows 下 CUDA bin 与 cuDNN bin 必须加进 PATH | 写进启动器；或改链 static CRT |
| **CT2 版本锁死** | 最新 ctranslate2 只支持 CUDA 12 + cuDNN 9 | CUDA11+cuDNN8 → `ctranslate2==3.24.0`；CUDA12+cuDNN8 → `4.4.0` |
| **端口占用** | TIME_WAIT 导致 `Address already in use`；`SO_REUSEADDR` 语义平台相关 | 启动前探端口 → 自动换端口（8787→8788…）→ 失败弹窗说明，**不弹 Python 栈** |
| **ffmpeg 缺失** | Windows 默认没有 ffmpeg | **随包分发 ffmpeg.exe**，启动器把其目录加进 PATH |

### 7.5 后端探测：不要「查询」，要「试跑」

任何 `is_available()` 都不等于「能用」（cuDNN 缺失、显存不足、算子不支持都会让查询通过但推理炸）。

```python
def build_backend(model_path, sample_wav):
    candidates = []
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            candidates.append(("cuda", "float16"))
            candidates.append(("cuda", "int8"))
    except Exception:
        pass
    candidates.append(("cpu", "int8"))          # 永远兜底

    for device, compute in candidates:
        try:
            m = WhisperModel(model_path, device=device, compute_type=compute)
            segs, _ = m.transcribe(sample_wav, beam_size=1)   # 跑 3 秒样本
            if not any(s.text.strip() for s in segs):
                raise RuntimeError("解码出空文本，判定不可用")
            return m
        except Exception as e:
            log.warning("后端 %s/%s 不可用，降级下一档: %s", device, compute, e)
    raise RuntimeError("所有后端均不可用")
```

**降级阶梯（Windows/Linux 通用）**：

| 档 | 条件 | 实现 |
|---|---|---|
| 1 | NVIDIA GPU 可用 | MLX-CUDA → 或 vLLM → 或 CT2 `cuda/float16` |
| 2 | GPU 在但 cuDNN 缺失/版本不符 | CT2 `cuda/int8` → sherpa-onnx CUDA EP |
| 3 | 无 GPU / 驱动过旧 / 显存不足 | sherpa-onnx ONNX `provider="cpu"`（int8） |
| 4 | CPU 太慢 | FunASR llama.cpp GGUF 单二进制 |

---

## 八、Linux 服务化

```ini
[Unit]
Description=Voice-to-Text Workbench
After=network-online.target
Wants=network-online.target

[Service]
Type=exec                 # systemd 官方："Typically, Type=exec is the better choice"
ExecStart=/opt/stt/.venv/bin/uvicorn app.server:app --host 127.0.0.1 --port 8787
WorkingDirectory=/opt/stt
Environment=PYTHONUTF8=1
Environment=OMP_NUM_THREADS=8
Restart=on-failure
RestartSec=5
TimeoutStopSec=30          # 否则优雅退出会被 SIGKILL

[Install]
WantedBy=multi-user.target
```

- `Type=notify` **不要用**（FastAPI 没接 sd_notify）。
- 配合本项目已修的「died 僵尸作业」：systemd 会自动拉起进程，
  但 `output/_status/*.json` 里的 died 记录不会自动清 → 启动脚本里先清旧状态。

---

## 九、依赖组织建议

```toml
[project]
requires-python = ">=3.11"
dependencies = ["fastapi", "uvicorn", "numpy", "python-multipart"]

[project.optional-dependencies]
macos        = ["mlx==0.32.4", "mlx-metal==0.32.4", "mlx-audio==0.6.3"]
linux-cuda   = ["mlx==0.32.4", "mlx-cuda-12==0.32.4", "mlx-audio==0.6.3"]
win-cuda     = ["mlx==0.32.4", "mlx-cuda-12==0.32.4", "mlx-audio==0.6.3"]  # 显式，不用 extra
win-cpu      = ["mlx==0.32.4", "mlx-cpu==0.32.4", "mlx-audio==0.6.3"]
cpu-onnx     = ["sherpa-onnx==2.0.1", "onnxruntime==1.31.1"]
cpu-gpu-onnx = ["sherpa-onnx==2.0.1", "onnxruntime-gpu==1.31.1"]
whisper      = ["faster-whisper==1.2.1"]
edge         = ["funasr==1.4.17"]
```

⚠️ **Windows 必须写 `mlx==0.32.4` + `mlx-cuda-12==0.32.4` 两个显式依赖**，
不能写 `mlx[cuda12]`（extra 被 Linux 标记挡住）。
⚠️ 平台探测做在**运行时**，不做在打包时。

---

## 十、最终建议：分五档交付

| # | 档位 | 引擎 | 模型 | 词级时间戳 | 状态 |
|---|---|---|---|---|---|
| 1 | macOS Apple Silicon | mlx-audio | `Qwen3-ASR-0.6B-8bit` + ForcedAligner | ✅ 独立对齐 | ✅ **已跑通（v1.0.0）** |
| 2 | Linux + N 卡 | MLX CUDA → vLLM 原生 | 同一个 Qwen3-ASR | ✅ | 待实测 |
| 3 | Windows + N 卡 | MLX CUDA（显式装）→ WSL2 跑 vLLM | 同上 | ✅ | 待实测 |
| 4 | A 卡 | FunASR 官方 Vulkan 运行时 | Paraformer-zh / SenseVoice | ⚠️ 降级 | 待实测 |
| 5 | 纯 CPU 兜底 | sherpa-onnx + Qwen3-ASR-0.6B int8 | 同模型 | ❌（要就换 Paraformer-zh） | ✅ 官方有 RTF |

**下一步应该只做一件事**：拿到一台 Linux + NVIDIA 机器，
跑 `pip install mlx[cuda12] mlx-audio` 后用现有的 `Qwen3-ASR-0.6B-8bit` 转 5 分钟中文，
和 Apple 路径逐字对比。**这一条验证通过，「保住模型换运行时」的路线就全线打通了**；
不通过就退回 vLLM 原生 Qwen3-ASR（官方 day-0 支持，代价是服务形态从库调用变成 HTTP）。

---

## 十一、明确未核实 / 待实测清单

| 项 | 状态 | 怎么验 |
|---|---|---|
| mlx-audio 的 Qwen3-ASR 在 **Linux CUDA / Windows CUDA / Windows CPU** 下能否跑通 | **未核实**（后端轮子已独立确认存在，但 mlx-audio 的算子覆盖未验证） | 一台 N 卡机器跑 5 分钟音频 |
| Qwen3-ASR 在 CUDA 上的**单流 RTF** | **未核实**（官方只给并发 128 时 2000× 吞吐） | 记录 1 小时音频 wall-clock |
| `qwen-asr` 包是否暴露 `hotwords`/`context` 参数 | **未核实**（已决定不装此包） | — |
| sherpa-onnx 的 Windows CUDA wheel 是否自带 cuDNN DLL | **未核实** | 目标机 `import sherpa_onnx` + 跑一次 GPU 推理 |
| sherpa-onnx Qwen3-ASR int8 的**模型体积** | **未核实** | 下载后看目录大小 |
| SenseVoice vs Whisper 的**具体 CER 数值** | **未核实**（官方表格是图片） | 用 `source/` 里的真实录音做 A/B |
| **A 卡上 Vulkan 的识别质量** | **有已知负面案例**（issue #2400） | 用真实录音抽检输出 |
| AMX 在 ASR 场景的实际加速倍数 | **未核实**（只有 LLM 场景的二手数据） | 找一台 Sapphire Rapids，OpenVINO vs fp32 对拍 |
| NVIDIA Riva 的 `boosted_lm_words` 热词参数 | **未核实**（来自第三方转载） | 回 NVIDIA 官方文档确认 |
| Parakeet TDT / Canary 1b 是否覆盖中文 | **存疑**（官方表里有 `parakeet-ctc-0.6b-zh-cn`，但有第三方资料称前两者不含中文） | 以官方表为准 + 实测 |
