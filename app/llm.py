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
from typing import Any, Dict, List, Optional, Sequence

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


def _user_config_path():
    """界面里保存的在线模型配置（存本地，不进 git）。"""
    return config.CACHE_DIR / "llm_config.json"


def load_user_config() -> Dict[str, str]:
    """读取界面保存的在线模型配置。"""
    p = _user_config_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {k: str(v) for k, v in data.items() if isinstance(v, (str, int, float))}
    except Exception:
        return {}


def save_user_config(data: Dict[str, Any]) -> Dict[str, str]:
    """保存界面填写的在线模型配置（Key 只落本地文件，权限 600）。"""
    clean: Dict[str, str] = {}
    for k in ("backend", "base", "model", "api_key"):
        v = str(data.get(k) or "").strip()
        if v:
            clean[k] = v
    # Key 留空表示「不修改」：界面不回显明文，避免误清已保存的 Key
    if "api_key" not in clean:
        old = load_user_config().get("api_key")
        if old:
            clean["api_key"] = old
    p = _user_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        p.chmod(0o600)
    except Exception:
        pass
    return clean


def clear_user_config() -> None:
    p = _user_config_path()
    if p.exists():
        p.unlink()


def describe_backends() -> List[Dict[str, Any]]:
    """可用后端清单：本地 + 预设 + 界面自定义。"""
    user = load_user_config()
    items: List[Dict[str, Any]] = [{
        "id": "local",
        "label": f"本地 {config.LLM_MODEL}（默认，离线）",
        "model": config.LLM_MODEL,
        "ready": True,
    }]
    for key, preset in config.LLM_PRESETS.items():
        key_env = preset.get("key_env") or ""
        ready = bool(user.get("api_key") if user.get("backend") == key
                     else os.environ.get(key_env) or config.LLM_API_KEY)
        items.append({
            "id": key,
            "label": preset.get("label") or key,
            "base": preset.get("base"),
            "model": preset.get("model"),
            "key_env": key_env or None,
            "ready": ready,
        })
    if user.get("base"):
        items.append({
            "id": "custom",
            "label": f"自定义 OpenAI 兼容：{user['base']}",
            "base": user.get("base"),
            "model": user.get("model") or "local-model",
            "ready": bool(user.get("api_key")),
            "key_env": None,
        })
    return items


def _resolve_online(name: str, override: Optional[Dict[str, str]] = None) -> tuple[str, str, str]:
    """解析在线后端的 (base, model, api_key)。

    优先级：override（界面未保存的测试值）→ 界面保存的用户配置 → 环境变量 → 预设默认值。
    """
    user = load_user_config()
    preset = config.LLM_PRESETS.get(name) or {}
    using_user = user.get("backend") == name
    ov = override or {}

    base = (ov.get("base") or "") or (user.get("base") if using_user else "") or config.LLM_API_BASE \
        or os.environ.get("STT_LLM_ENDPOINT", "").strip() or preset.get("base", "")
    model = (ov.get("model") or "") or (user.get("model") if using_user else "") or config.LLM_API_MODEL \
        or preset.get("model", "local-model")
    key_env = preset.get("key_env") or ""
    key = (ov.get("api_key") or "") or (user.get("api_key") if using_user else "") or config.LLM_API_KEY \
        or (os.environ.get(key_env, "") if key_env else "")
    return base.rstrip("/"), model, key


def _build_client(backend: Optional[str] = None, override: Optional[Dict[str, str]] = None):
    """返回 (generate_fn, 说明)。generate_fn(prompt, system) -> str

    backend:
        None/local  → mlx-lm 本地推理（默认）
        openai      → 通用 OpenAI 兼容接口
        minimax / deepseek / … → 预设的 OpenAI 兼容服务
    界面里保存的配置优先级最高（界面配置 → 环境变量 → 预设默认值）。
    override 用于「测试连接」：直接用界面上尚未保存的地址/模型/Key 试连。
    """
    user = load_user_config()
    name = (backend or user.get("backend") or config.LLM_BACKEND or "local").strip().lower()

    if name in ("local", "mlx", "mlx-lm"):
        return _build_local_client()

    preset = config.LLM_PRESETS.get(name)
    base, model, api_key = _resolve_online(name, override=override)
    if not base:
        raise RuntimeError(
            f"未知后端 {name}；可用：local、openai，或 {', '.join(config.LLM_PRESETS)}"
        )
    if not api_key and not _is_local_endpoint(base):
        raise RuntimeError(
            f"后端 {name} 需要 API Key：请在界面「模型设置」里填写，"
            f"或设置 {preset.get('key_env') if preset else 'STT_LLM_API_KEY'} 环境变量"
        )
    return _openai_client(base, api_key=api_key, model=model), f"openai:{name}:{model}"


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
            # 先清掉 mlx 分配器缓存：转写用的 ASR 模型刚释放，缓存不回收的话
            # 16GB 机器上再加载 8B 润色模型会分配失败。
            try:
                import mlx.core as mx  # type: ignore

                mx.clear_cache()
            except Exception:  # pragma: no cover
                pass
            try:
                model, tokenizer = mlx_load(config.local_model(config.LLM_MODEL))
            except Exception as exc:
                raise RuntimeError(
                    f"加载本地润色模型失败：{exc}。"
                    "若提示内存不足，多半是转写模型还占着显存——"
                    "重启服务后再润色，或改用网络 API（界面「⚙️ 模型设置」）。"
                ) from exc
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


def test_connection(backend: Optional[str] = None,
                   override: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """连通性自检：发一个极小的请求，确认后端与 Key 可用。

    override 传入界面尚未保存的 base/model/api_key，可直接试连。
    """
    name = (backend or load_user_config().get("backend") or config.LLM_BACKEND or "local").lower()
    if name in ("local", "mlx", "mlx-lm"):
        return {"ok": False, "backend": "local",
                "message": "本地后端需要 GPU（Metal）；当前环境不可用时请改用网络 API"}
    try:
        fn, info = _build_client(name, override=override)
    except Exception as exc:
        return {"ok": False, "backend": name, "message": str(exc)}
    try:
        out = fn("回复两个字：好的", "只回复「好的」两个字。", max_tokens=16, temperature=0.1)
    except Exception as exc:
        return {"ok": False, "backend": name, "message": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "backend": name, "info": info,
            "message": f"连通正常，模型回应：{(out or '')[:40]}"}


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
