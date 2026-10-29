# 跨平台 ASR 改造计划（代码侧设计）

> 分支：`research/asr-cross-platform`
> 基线：`v1.0.0`（Apple Silicon 本地版收口）
> 选型结论见另一篇《跨平台 ASR 方案（调研报告）》。

## 结论先行

1. **不用重写，接口已经够小。** `app/asr.py` 的 `BaseEngine` 只有三个方法
   （`load` / `transcribe` / `align`），加新引擎就是加一个类，不需要动 pipeline。
2. **真正的阻塞点只有三处**，其余都是配置问题：
   - `app/asr.py` + `app/asr_whisper.py` 里 `from mlx_audio.stt import load` → 只能 Apple Silicon；
   - `setup.sh` 把 `mlx-audio mlx-lm` 写死进依赖，且没有 `requirements.txt`；
   - `app/audio.py` 裸调 `ffmpeg` 命令（依赖 PATH），Windows 上默认没有。
3. **LLM 侧已经不是问题。** `app/llm.py` 的 `_build_client()` 本来就支持任意
   OpenAI 兼容服务（`STT_LLM_BACKEND=custom` + `STT_LLM_API_BASE`），
   本机已在用内网 `192.168.162.100:2353`。跨平台只需要换 ASR。
4. **`app/monitor.py` 是 macOS 专属**（`ioreg` / `top -l 1` / `vm_stat`），
   在 Linux/Windows 上会整块退化成「—」。不影响功能，但仪表盘会瞎。
5. **词级时间戳是硬约束**，不是加分项。`app/pipeline.py` 依赖 `asr.align()`
   给出词级时间戳来切句；拿不到就退化到 `app/align.py` 的兜底——
   把**整块时间区间按字符数比例**摊给各句（等于假设语速均匀），
   30 秒一块逐块摊下来，绝对位置会明显漂。**选引擎时自带时间戳优先。**

---

## 一、现状盘点（均为代码事实，已逐个核对）

### 1.1 ASR 抽象层 `app/asr.py`

```python
class BaseEngine:
    def load(self) -> None: ...
    def transcribe(self, wav_path: Path, offset: float = 0.0) -> Segment: ...
    def align(self, wav_path: Path, text: str, offset: float = 0.0) -> Optional[List[Dict]]: ...
```

- `Segment` = `{start, end, text, words?}`。
- `align()` 返回 `[{text, start, end}]`，时间是相对整段音频的绝对秒。
- `build_engine(engine)` 目前只认 `qwen3` / `qwen3-hq` / `whisper` 三档，
  且两个实现都走 `mlx_audio.stt.load()`。

### 1.2 调用链

| 位置 | 事实 |
|---|---|
| `app/server.py` `/api/transcribe` | 接受 `engine` / `align` / `hotwords` / `force` / `rules` |
| `app/pipeline.py:281` | `asr = build_engine(engine, language=…, use_aligner=align, hotwords=…)` |
| `app/pipeline.py:324` | `words = asr.align(chunk_wav, text, offset=0.0) if align else None` |
| `app/pipeline.py:325` | `segs = self._split_into_segments(text, words, start, end)` → `app/align.py` |
| `app/pipeline.py` 块级处理 | 整段音频先按静音切块，**逐块转写**，块边界落盘可断点续跑 |

### 1.3 平台耦合点清单

| 文件 | 平台假设 | 后果 |
|---|---|---|
| `app/asr.py` | `from mlx_audio.stt import load` | Linux/Windows 直接 ImportError |
| `app/asr_whisper.py` | 同上 | 同上 |
| `app/audio.py` | 裸调 `ffmpeg`（依赖 PATH），用 `silencedetect` 滤镜 | Windows 默认无 ffmpeg → 切块/静音检测全废 |
| `app/monitor.py` | `ioreg -c AGXAccelerator`、`top -l 1`、`vm_stat` | 非 macOS 仪表盘硬件栏全「—」 |
| `app/llm.py` | `_build_local_client()` → `from mlx_lm import load` | 仅影响「纯本地 LLM」这条可选路径；走 OpenAI 兼容服务即可跨平台 |
| `setup.sh` | bash + 写死 `mlx-audio mlx-lm` | Windows 装不了 |
| `run.sh` | bash | Windows 装不了 |
| `scripts/fetch_models.py` | 只下 `mlx-community/*`（MLX 量化 safetensors） | 其他格式模型下不了 |

### 1.4 配置 `app/config.py`

| 变量 | 默认值 | 说明 |
|---|---|---|
| `STT_ASR_MODEL` | `mlx-community/Qwen3-ASR-0.6B-8bit` | MLX 量化格式 |
| `STT_ASR_MODEL_HQ` | `mlx-community/Qwen3-ASR-1.7B-8bit` | 高精度档 |
| `STT_ALIGNER_MODEL` | `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` | 词级强制对齐（可选） |
| `STT_LLM_MODEL` | `mlx-community/Qwen3-8B-4bit` | 本地 LLM（可换成 HTTP 服务） |
| `STT_APP_BUILD` | `2026-10-28-no-think-kb` | 构建标记，用来确认跑着的是不是最新代码 |

`config.local_model()` 已支持「模型已下到 `models/` 就用本地路径，否则回落到仓库 ID」，
这个机制与平台无关，**新引擎直接复用**。

---

## 二、目标架构

### 2.1 引擎注册表（新增 `app/asr_registry.py`）

把「有哪些引擎、每个引擎在什么平台上能跑、支持什么特性」从代码里抽成声明式数据：

```python
@dataclass(frozen=True)
class EngineSpec:
    key: str                       # "qwen3" / "faster-whisper" / "funasr" / "sherpa-onnx" / "whisper-cpp"
    label: str
    platforms: frozenset[str]      # {"darwin", "linux", "win32"}
    backends: frozenset[str]       # {"mlx", "cuda", "rocm", "vulkan", "directml", "cpu"}
    word_timestamps: bool           # 能否给出词级时间戳
    hotwords: bool                 # 是否支持热词
    pip_extra: str                 # "cuda" / "cpu" / "macos" …
    model_repo: str                # 模型仓库（按引擎不同）
```

`build_engine()` 改为查表 + 延迟导入：**没装的引擎不应该让整个模块 import 失败**，
只有真被选中时才 import 它的依赖。

### 2.2 加速后端探测与自动降级（新增 `app/device.py`）

**原则：不要「查询」，要「试跑」。** 任何 `is_available()` 都不等于「能用」——
cuDNN 缺失、显存不足、算子不支持，都会让查询通过但推理炸。
正确做法是**用 3 秒样本真跑一遍**，拿到非空文本才算这一档可用。

| 平台 | 优先级 | 探测手段 |
|---|---|---|
| macOS | mlx | `import mlx.core` 成功 + 试跑 |
| Linux/Windows + NVIDIA | mlx-cuda → vLLM → sherpa-onnx CUDA | 逐档**试跑**，失败即降级 |
| Windows + AMD/Intel | FunASR Vulkan → whisper.cpp Vulkan → CPU | 逐档试跑 |
| Linux + AMD | FunASR Vulkan → PyTorch ROCm → whisper.cpp HIP → CPU | 逐档试跑 |
| 兜底 | CPU | 总是可用 |

⚠️ **DirectML 不进降级链**：它拿不到词级时间戳（通用 whisper→ONNX 导出不含 cross-attention），
本项目「词级时间戳」是硬约束。只有用户**显式关闭时间戳**时才允许把
`sherpa-onnx + DirectML` 作为候选，并在界面上标注质量降级。

⚠️ **Linux + N 卡第一档首选 vLLM 而不是 CT2**：CT2 在 NVIDIA 上只支持 CUDA 12 + cuDNN 9
（版本锁死），而 vLLM / MLX CUDA 直接跑 Qwen3-ASR 原生权重，能保住模型与对齐器。

**降级要可见**：实际用的后端写进作业状态（`output/_status/*.json` 的 `meta`），
仪表盘上标出来。不能出现「装了个 CUDA 版但其实跑在 CPU 上，用户不知道」。
失败时**降级并给中文提示**，不弹 Python 栈给最终用户。

### 2.3 时间戳契约

`align()` 返回 `None` 是合法值，但 pipeline 会退化成按字数估时。
因此每个候选引擎都必须回答：**词级时间戳从哪来？**

| 来源 | 代表引擎 | 跨平台成本 |
|---|---|---|
| 模型自带 segment/word 时间戳 | Whisper 系 | 低 |
| 模型自带 timestamp token | Paraformer / SenseVoice | 低 |
| 额外跑强制对齐器 | Qwen3-ForcedAligner | 高（当前只有 MLX） |
| 无 → 按字数估时 | — | 零成本但长稿会漂，**不可作为交付质量** |

**决策：优先选自带时间戳的引擎；把「无对齐器」标记为质量降级而不是静默兜底。**

把候选引擎逐个对上（数据来自《跨平台 ASR 方案》）：

| 候选引擎 | 词级时间戳来源 | 质量 |
|---|---|---|
| MLX Qwen3-ASR + Qwen3-ForcedAligner（现状） | 独立强制对齐器 | ✅ 最优（AAS 中文 33.1ms） |
| vLLM 原生 Qwen3-ASR | 官方 `qwen3_asr_forced_aligner.py` | ✅ 同等 |
| transformers Qwen3-ASR | `AutoModelForTokenClassification` | ✅ 同等 |
| sherpa-onnx + Qwen3-ASR int8 | ❌ `timestamps` 为空 | ⚠️ 只能按字数估时 |
| sherpa-onnx + Paraformer-zh int8 | ✅ 模型自带 timestamps | ✅ 可交付 |
| FunASR SenseVoice | ❌ `timestamps` 为空（靠 VAD 给句级） | ⚠️ |
| faster-whisper | ✅ `word_timestamps=True` | ✅ 但**无热词 API** |
| whisper.cpp | ⚠️ `-ml 1`（官方标 experimental） | ⚠️ |

### 2.4 依赖拆分

现在是 `setup.sh` 里一行写死，且**没有 `requirements.txt`**。改为 `pyproject.toml` + extras：

```toml
[project]
requires-python = ">=3.11"
dependencies = ["fastapi", "uvicorn", "numpy", "python-multipart"]

[project.optional-dependencies]
macos        = ["mlx==0.32.4", "mlx-metal==0.32.4", "mlx-audio==0.6.3", "mlx-lm"]
linux-cuda   = ["mlx==0.32.4", "mlx-cuda-12==0.32.4", "mlx-audio==0.6.3"]
win-cuda     = ["mlx==0.32.4", "mlx-cuda-12==0.32.4", "mlx-audio==0.6.3"]  # 显式，不写 extra
win-cpu      = ["mlx==0.32.4", "mlx-cpu==0.32.4", "mlx-audio==0.6.3"]
cpu-onnx     = ["sherpa-onnx==2.0.1", "onnxruntime==1.31.1"]
cpu-gpu-onnx = ["sherpa-onnx==2.0.1", "onnxruntime-gpu==1.31.1"]
edge         = ["funasr==1.4.17"]
whisper      = ["faster-whisper==1.2.1"]
```

`setup.sh` 保留给 macOS/Linux，另加 `setup.ps1` 给 Windows。

⚠️ **两个必须写进 README 的坑**：

1. **Windows 上不能用 extra**：MLX 的 extras 全部带 `platform_system == "Linux"` 标记，
   `pip install mlx[cuda]` 在 Windows 上**不会**装上后端包，只会装到 0.6MB 的空壳 core。
   必须显式写 `mlx==0.32.4` + `mlx-cuda-12==0.32.4`。
2. **MLX 的 CUDA 后端要求 glibc ≥ 2.35**（= Ubuntu 22.04+）；
   CentOS 7 / RHEL 8（glibc 2.28）**装不了 MLX 的任何后端**，这类机器只能走 vLLM / ONNX。

### 2.5 模型获取 `scripts/fetch_models.py`

现在只认 `mlx-community/*`。要按引擎分源：

| 引擎 | 模型来源 | 格式 |
|---|---|---|
| qwen3 / whisper（mlx） | HF/ModelScope `mlx-community/*` | MLX 量化 safetensors |
| qwen3（vLLM / transformers） | HF `Qwen/Qwen3-ASR-0.6B-hf` / `-1.7B-hf` | bf16 safetensors（官方权重） |
| faster-whisper | HF `Systran/faster-whisper-*` | CTranslate2 |
| funasr | ModelScope `iic/*` | torch / onnx |
| funasr 离线二进制 | FunASR 官方 release（`windows-x64-{cuda,vulkan,avx2}.zip` / linux 同款） | GGUF（sensevoice + fsmn-vad） |
| sherpa-onnx | 官方 release / HF | onnx（Qwen3-ASR-0.6B int8 / Paraformer-zh / SenseVoice） |

**离线安装**也要支持：企业内网环境不能现下几个 GB 的模型，
需要 `fetch_models.py --offline <本地目录>` 或直接拷 `models/` 目录过去。

---

## 三、分阶段路线

| 阶段 | 内容 | 产出 | 风险 |
|---|---|---|---|
| 0 | 调研 + 设计（本分支） | 本篇 + 《跨平台 ASR 方案》 | 低 |
| 1 | 抽象层重构 | `asr_registry.py` + `device.py`；**现有 mlx 行为零变化** | 低 |
| 2 | **保住模型换运行时**：Linux/N 卡跑 MLX CUDA 或 vLLM 原生 Qwen3-ASR | 新增 `qwen3-cuda` / `qwen3-vllm` 引擎 + `linux-cuda` extras | 中（算子覆盖未实测） |
| 3 | **兜底与 A 卡**：sherpa-onnx（CPU/CUDA）+ FunASR Vulkan | ONNX 引擎 + `cpu-onnx` extras + Vulkan 分发 | 中（A 卡时间戳降级） |
| 4 | Windows 打包 + 离线安装 | `setup.ps1` / onedir 打包 / ffmpeg 内置 / 模型外置 | 中高（杀软、MAX_PATH、GBK） |
| 5 | 回归验证 | 同一批音频的多引擎 CER 对比表 | — |

**阶段 1 的验收标准很硬**：`v1.0.0` 的 11 篇产物必须**一字节不变**，
`verify_all` 仍 11/11，macOS 上的行为与现在完全一致。

**阶段顺序的依据**（来自《跨平台 ASR 方案》）：先做阶段 2 而不是阶段 3，
因为「保住模型换运行时」一旦验证通过，Windows N 卡与 Linux N 卡可以**复用同一套模型与同一份代码**，
投入产出比最高；而 sherpa-onnx 虽然覆盖面最大，但要**放弃 Qwen3-ASR**（`timestamps` 为空），
代价是词级时间戳降级，属于不得已的下一步。

---

## 四、验收标准（阶段 2 之后）

1. **功能**：同一段 1 小时中文录音，在 N 卡 / A 卡 / CPU 上都能产出完整逐字稿。
2. **质量**：三种硬件上逐字稿的 **CER 差异可控**（同一引擎换后端，差异应远小于换引擎的差异）。
3. **时间轴**：词级时间戳单调、不越界、不出现负值；无对齐器时必须在界面上标注降级。
4. **回归**：`verify_all` 11/11；macOS 现有产物不变。
5. **可交付**：Windows 上从零到跑起来 ≤ 30 分钟（含模型下载），全离线安装可行。

---

## 五、待研究确认的点

调研阶段已回答的问题（详见《跨平台 ASR 方案》）：

- Qwen3-ASR 能不能脱离 MLX 跑 —— **能**。vLLM 0.32.0 原生含 `qwen3_asr.py` +
  `qwen3_asr_forced_aligner.py`（官方 day-0 支持）；transformers ≥5.13.0 有官方 `-hf` 权重；
  sherpa-onnx 提供 Qwen3-ASR-0.6B int8 ONNX。
- A 卡上哪条路是死路 —— **faster-whisper/CT2 的 GPU 只支持 NVIDIA**（CT2 官方硬件页白名单），
  WhisperX 连带死；**DirectML 拿不到词级时间戳**，不满足本项目硬约束。
  A 卡可用的是 **FunASR 官方 Vulkan 运行时**（Windows/Linux 都有官方 zip）与 whisper.cpp HIP。
- CPU 兜底的 RTF 量级 —— sherpa-onnx + Qwen3-ASR-0.6B int8 **RTF 0.077~0.103**；
  Paraformer-zh **RTF 0.073** 且**自带 timestamps + 原生热词**（本项目热词刚需）。
- Windows 上 ffmpeg 与 cuDNN 的分发方式 —— ffmpeg **随包分发**；
  cuDNN **pip 装不上（Linux 限定）**，需下载官方 DLL 并加进 `PATH`。

仍需**实测**才能拍板的问题：

| 问题 | 为什么阻塞 | 怎么验 |
|---|---|---|
| mlx-audio 的 Qwen3-ASR 在 Linux/Windows CUDA 下能否跑通 | 决定阶段 2 走 MLX CUDA 还是 vLLM | 一台 N 卡机器转 5 分钟中文，与 Apple 路径逐字对比 |
| Qwen3-ASR 在 CUDA 上的单流 RTF | 决定是否需要 batch | 记录 1 小时音频 wall-clock |
| sherpa-onnx 的 Windows CUDA wheel 是否自带 cuDNN DLL | 决定 Windows N 卡能否免装 | 目标机 `import sherpa_onnx` + 跑一次 GPU 推理 |
| SenseVoice vs Whisper 的具体 CER | 决定 A 卡/CPU 兜底能不能上生产 | 用 `source/` 里的真实录音做 A/B |
| A 卡上 Vulkan 的识别质量 | 已知负面案例 issue #2400 | 真实录音抽检输出 |
| AMX 在 ASR 场景的实际加速倍数 | 决定 Intel CPU 档是否值得投入 | Sapphire Rapids + OpenVINO vs fp32 对拍 |
