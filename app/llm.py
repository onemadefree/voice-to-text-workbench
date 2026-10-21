"""本地 LLM 后处理：同音错字纠正、口语去除、书面化润色、知识库提炼。

后端优先用 mlx-lm（Apple Silicon 原生），也可指向任意 OpenAI 兼容服务
（LM Studio / llama.cpp server / Ollama），通过环境变量切换。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence

from . import config

log = logging.getLogger("stt.llm")

POLISH_SYSTEM = """你是一名中文文稿编辑。你会收到一段演讲的原始转写文本，需要你把它整理成书面稿。

严格要求：
1. 只做纠错与润色，不得增加原文没有的信息，不得改变讲者的观点、数字和结论。
2. 修正同音字、错别字、专有名词错误（结合上下文判断；中文数字年份/百分比/数量要转成阿拉伯数字，如"二零二五年"→"2025年"；英文专有名词按通行写法，如 "R and D" → "R&D"）。
3. 去掉口语表达：语气词（嗯、呃、啊、那个、然后就是）、口头禅、重复、自我修正的痕迹。
4. 保留讲者原始的段落意图，但允许合并碎句、补全标点，使句子通顺可读。
5. 保留其中的专业术语、产品名、人名、地名，不要意译或改写专有名词。
6. 严格保持条目数量与顺序：输入 N 条就必须输出 N 条，序号一一对应，不得遗漏、合并、拆分或新增。

输出格式（严格遵守）：每行一个 JSON 对象，不要加任何其他文字或代码块标记：
{"i": 输入序号, "t": "改写后的文本"}
"""

# 备用格式：部分模型更擅长按编号文本输出
POLISH_SYSTEM_FALLBACK = POLISH_SYSTEM.replace(
    '输出格式（严格遵守）：每行一个 JSON 对象，不要加任何其他文字或代码块标记：\n{"i": 输入序号, "t": "改写后的文本"}\n',
    "输出格式（严格遵守）：每行一条，格式为「序号. 改写后的文本」，序号从 1 开始连续递增，不得遗漏或合并。\n",
)

KB_SYSTEM = """你是一名内容编辑。你会收到一段演讲的整理稿，请提炼成结构化的知识库资料。

输出严格使用以下 Markdown 结构，不要有多余说明：
# 标题
## 摘要
（3-5 句话概括核心观点）
## 要点
- （逐条列出，每条一句话）
## 术语表
- 术语：解释
## 待确认
- （原文含糊或存疑的点，没有则写"无"）
"""


def _build_client(backend: Optional[str] = None):
    """返回 (generate_fn, 说明)。generate_fn(prompt, system) -> str

    backend:
        None/local  → mlx-lm 本地推理（默认）
        openai      → 通用 OpenAI 兼容接口（用 STT_LLM_API_* 配置）
        minimax / deepseek / … → 预设的 OpenAI 兼容服务
    """
    name = (backend or config.LLM_BACKEND or "local").strip().lower()

    if name in ("local", "mlx", "mlx-lm"):
        return _build_local_client()

    preset = config.LLM_PRESETS.get(name)
    # 自定义 base 优先：配了 STT_LLM_API_BASE / STT_LLM_ENDPOINT 就用它，
    # 此时模型名沿用预设或 STT_LLM_API_MODEL。
    custom_base = (config.LLM_API_BASE or os.environ.get("STT_LLM_ENDPOINT", "").strip()).rstrip("/")
    if custom_base:
        model = config.LLM_API_MODEL or (preset or {}).get("model") or "local-model"
        api_key = config.LLM_API_KEY or os.environ.get("STT_LLM_API_KEY", "")
        if not api_key and not _is_local_endpoint(custom_base):
            raise RuntimeError(
                f"后端 {name} 需要 API Key：请设置 STT_LLM_API_KEY 环境变量"
            )
        return _openai_client(custom_base, api_key=api_key, model=model), f"openai:{custom_base}:{model}"

    if preset:
        base = preset["base"]
        model = config.LLM_API_MODEL or preset["model"]
        key_env = preset.get("key_env") or ""
        api_key = config.LLM_API_KEY or (os.environ.get(key_env, "") if key_env else "")
        if not api_key and not _is_local_endpoint(base):
            raise RuntimeError(
                f"后端 {name} 需要 API Key：请设置 {key_env or 'STT_LLM_API_KEY'} 环境变量"
            )
        return _openai_client(base, api_key=api_key, model=model), f"openai:{name}:{model}"

    raise RuntimeError(
        f"未知后端 {name}；可用：local、openai，或 {', '.join(config.LLM_PRESETS)}"
    )


def _is_local_endpoint(base: str) -> bool:
    """本机服务（LM Studio / Ollama 等）不需要 API Key。"""
    return any(h in base for h in ("127.0.0.1", "localhost", "0.0.0.0", "[::1]"))


def _build_local_client():
    """mlx-lm 本地推理。"""
    try:
        from mlx_lm import load as mlx_load, generate as mlx_generate  # type: ignore
        from mlx_lm.sample_utils import make_sampler  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "未安装 mlx-lm，也没有配置 STT_LLM_ENDPOINT；请运行 setup.sh 安装依赖"
        ) from exc

    state: Dict[str, object] = {}

    def _chat_prompt(tokenizer, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        apply = getattr(tokenizer, "apply_chat_template", None)
        if apply is not None:
            for kwargs in ({"enable_thinking": False}, {}):
                try:
                    return apply(messages, tokenize=False, add_generation_prompt=True, **kwargs)
                except TypeError:
                    continue
                except Exception:
                    break
        # 兜底：Qwen 风格的模板
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate(prompt: str, system: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
        if state.get("model") is None:
            log.info("加载 LLM: %s", config.LLM_MODEL)
            model, tokenizer = mlx_load(config.local_model(config.LLM_MODEL))
            state["model"] = model
            state["tokenizer"] = tokenizer
        sampler = make_sampler(temp=temperature)
        text = mlx_generate(
            state["model"],
            state["tokenizer"],
            prompt=_chat_prompt(state["tokenizer"], system, prompt),
            max_tokens=max_tokens,
            sampler=sampler,
        )
        return (text or "").strip()

    return generate, f"mlx-lm:{config.LLM_MODEL}"


def _openai_client(base: str, api_key: str = "", model: str = "local-model"):
    """任意 OpenAI 兼容服务（MiniMax / DeepSeek / 通义 / Kimi / 本地 server）。

    只用标准库 urllib 实现，不引入额外依赖；带指数退避重试。
    """
    base_url = base.rstrip("/")
    if not base_url.endswith("/v1") and "/v1" not in base_url:
        base_url += "/v1"
    url = base_url + "/chat/completions"
    headers = {"Content-Type": "application/json", "User-Agent": "stt-workbench"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    retries = max(1, config.LLM_API_RETRIES)

    def generate(prompt: str, system: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }, ensure_ascii=False).encode("utf-8")
        last = ""
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=config.LLM_API_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode("utf-8", "ignore"))
                msg = (data.get("choices") or [{}])[0].get("message") or {}
                # 推理型模型（如 MiniMax-M1/M2）正文可能落在 reasoning_content
                text = (msg.get("content") or "").strip()
                if not text:
                    text = (msg.get("reasoning_content") or "").strip()
                if text:
                    return text
                last = "空响应"
            except Exception as exc:  # 网络抖动/限流 → 退避重试
                last = f"{type(exc).__name__}: {exc}"
                log.warning("LLM 调用失败（第 %d 次）: %s", attempt, last[:200])
            if attempt < retries:
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"LLM 调用失败：{last}")

    return generate


class LocalLLM:
    """线程安全的 LLM 包装（本地 mlx-lm 或 OpenAI 兼容网络 API）。"""

    def __init__(self, backend: Optional[str] = None) -> None:
        self._generate = None
        self._info = ""
        self._backend = backend
        self._lock = threading.Lock()

    @property
    def info(self) -> str:
        if not self._info:
            self._ensure()
        return self._info

    def _ensure(self):
        if self._generate is None:
            with self._lock:
                if self._generate is None:
                    self._generate, self._info = _build_client(self._backend)

    def complete(self, prompt: str, system: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
        self._ensure()
        return self._generate(prompt, system, max_tokens=max_tokens, temperature=temperature)


# ---------------------------------------------------------------- 批量润色

_NUM_RE = re.compile(r"^(\d{1,3})\.\s*(.*)$", re.S)


def polish_segments(
    llm: LocalLLM,
    segments: Sequence[Dict],
    context_head: str = "",
    batch_size: int = 16,
    max_chars: int = 2200,
    align_threshold: float = 0.30,
) -> List[str]:
    """对原始文本批量润色，返回与 segments 等长的改写文本列表。

    两道保险：
    1. 输出用带序号的 JSON 行，模型漏行也不会让后续条目整体错位；
    2. 每条结果做与原文的字符相似度校验，明显不对应的退回原文。
    """
    out: List[str] = [""] * len(segments)
    batch: List[int] = []
    buf_len = 0

    def flush(idxs: List[int]) -> None:
        if not idxs:
            return
        lines = []
        for i in idxs:
            lines.append(f"{i + 1}. {segments[i].get('raw') or segments[i].get('text') or ''}")
        body = "\n".join(lines)
        head = f"上文提要：{context_head}\n\n" if context_head else ""
        prompt = (
            f"{head}请逐条改写下面 {len(idxs)} 条转写文本，"
            f"严格保持 {len(idxs)} 条、顺序与序号不变。\n\n{body}"
        )
        raw_out = ""
        for system in (POLISH_SYSTEM, POLISH_SYSTEM_FALLBACK):
            try:
                raw_out = llm.complete(prompt, system, max_tokens=max_tokens_for(idxs), temperature=0.25)
            except Exception as exc:
                log.warning("润色批次失败: %s", exc)
                raw_out = ""
            if raw_out.strip():
                break
        mapping = _parse_indexed(raw_out, len(idxs))
        for pos, i in enumerate(idxs):
            src = (segments[i].get("raw") or segments[i].get("text") or "").strip()
            cand = (mapping.get(pos) or "").strip()
            if not cand:
                out[i] = src
                continue
            if not _aligned(src, cand, align_threshold):
                log.info("润色结果与原文不匹配（第 %d 条），保留原文", pos + 1)
                out[i] = src
                continue
            out[i] = cand

    for idx, seg in enumerate(segments):
        text_len = len(seg.get("raw") or seg.get("text") or "")
        if batch and (len(batch) >= batch_size or buf_len + text_len > max_chars):
            flush(batch)
            batch, buf_len = [], 0
        batch.append(idx)
        buf_len += text_len
    flush(batch)
    return out


def _aligned(src: str, cand: str, threshold: float = 0.30) -> bool:
    """校验润色结果确实对应原文（字符相似度过低说明条目错位或模型跑偏）。"""
    if not src or not cand:
        return False
    if src == cand:
        return True
    import difflib

    return difflib.SequenceMatcher(None, src, cand).ratio() >= threshold


def max_tokens_for(idxs: Sequence[int]) -> int:
    return max(1024, min(8192, len(idxs) * 320))


def _parse_indexed(raw: str, expect: int) -> Dict[int, str]:
    """解析带序号的润色输出，返回 {0 基序号: 文本}。

    优先按 JSON 行解析（带显式 i，不会因漏行导致整体错位），
    失败再退回「序号. 文本」纯文本格式。
    """
    if not raw:
        return {}
    text = raw.strip()

    # 去掉可能出现的 ```json 代码块包裹
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    result: Dict[int, str] = {}

    # ---- 方式一：整个 JSON 数组
    if text.startswith("["):
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                for idx, item in enumerate(arr):
                    if isinstance(item, dict):
                        i = item.get("i")
                        t = item.get("t") or item.get("text")
                        if t is not None and i is not None:
                            result[int(i) - 1] = str(t)
                        elif t is not None:
                            result[idx] = str(t)
                if result:
                    return result
        except Exception:
            pass

    # ---- 方式二：JSON 行
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        i = obj.get("i", obj.get("index"))
        t = obj.get("t", obj.get("text"))
        if t is None:
            continue
        if i is None:
            i = len(result) + 1
        try:
            result[int(i) - 1] = str(t)
        except (TypeError, ValueError):
            continue
    if result:
        return result

    # ---- 方式三：纯文本「序号. 内容」
    return _parse_numbered(text, expect)


def _parse_numbered(raw: str, expect: int) -> Dict[int, str]:
    """解析「1. xxx」编号格式，返回 {序号(0基): 文本}。"""
    result: Dict[int, str] = {}
    if not raw:
        return result
    current = None
    buf: List[str] = []
    for line in raw.splitlines():
        m = _NUM_RE.match(line.strip())
        if m:
            if current is not None:
                result[current] = "\n".join(buf).strip()
            current = int(m.group(1)) - 1
            buf = [m.group(2).strip()]
        elif current is not None:
            buf.append(line.strip())
    if current is not None:
        result[current] = "\n".join(buf).strip()

    # 兜底：没有编号时按空行/整段切分
    if not result:
        parts = [p.strip() for p in re.split(r"\n{1,}", raw) if p.strip()]
        for i, p in enumerate(parts[:expect]):
            result[i] = p
    # 编号异常时按顺序回填
    if result and sorted(result) != list(range(expect)):
        ordered = [result[k] for k in sorted(result)]
        result = {i: v for i, v in enumerate(ordered)}
    return result


# ---------------------------------------------------------------- 知识库提炼

CHUNK_SYSTEM = """你是一名中文文稿编辑。你会收到一段演讲稿的中间部分，请提取其中值得保留的信息。

输出要点，每行一条，不要标题、不要解释：
- （事实、数据、观点、案例、结论；保留具体数字与人名地名）"""


def build_knowledge_base(llm: LocalLLM, full_text: str, title: str = "") -> str:
    """长稿走「分段提炼 → 汇总」两段式，避免一次性超长输入。"""
    if not full_text.strip():
        return ""

    # ---- 第一步：分段提炼
    size = 2600
    parts = [full_text[i:i + size] for i in range(0, len(full_text), size)]
    notes: List[str] = []
    for idx, part in enumerate(parts):
        try:
            out = llm.complete(
                f"这是演讲稿的第 {idx + 1}/{len(parts)} 部分：\n\n{part}",
                CHUNK_SYSTEM,
                max_tokens=1400,
                temperature=0.2,
            )
        except Exception as exc:
            log.warning("分段提炼失败 (%d/%d): %s", idx + 1, len(parts), exc)
            continue
        if out.strip():
            notes.append(out.strip())
    if not notes:
        return ""

    merged = "\n".join(notes)
    # 合并后仍过长时再压一轮
    if len(merged) > 9000:
        try:
            merged = llm.complete(
                f"以下是演讲稿各部分的要点，请去重合并：\n\n{merged}",
                CHUNK_SYSTEM,
                max_tokens=2600,
                temperature=0.2,
            )
        except Exception:
            pass

    # ---- 第二步：结构化汇总
    title_hint = f"标题：{title}\n\n" if title else ""
    prompt = (
        f"{title_hint}以下是演讲整理稿及要点提炼，请产出结构化的知识库资料。\n\n"
        f"【整理稿】\n{full_text[:6000]}\n\n"
        f"【要点提炼】\n{merged}\n"
    )
    try:
        return llm.complete(prompt, KB_SYSTEM, max_tokens=4096, temperature=0.2)
    except Exception as exc:
        log.warning("知识库生成失败: %s", exc)
        return ""
