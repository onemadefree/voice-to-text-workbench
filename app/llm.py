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
7. 四位年份必须完整保留：像"二零二零年""二零二二年"这类中文数字年份，转成阿拉伯数字时必须是 2020年、2022年，
   不能写成"二零年"（丢位），也不能写成"2020-20年"。原文里的所有数字、百分比、金额、日期、编号一律不得改动或省略。
8. 年份的还原要结合上下文判断：语音识别有时把年份写坏，例如"二五二五年""七六二五年""二六二六年"，
   逐位读是 2525/7625/2626，但讲者说的其实是 2025年/2026年——要结合上下文还原成正确的阿拉伯数字年份，
   绝不能按字面转成 2525年、7625年这种不存在的年份。反过来，真实存在的历史年份
   （如"一六七九年""一八四七年""一八四八年"＝1679/1847/1848）要如实保留成 1679年、1847年、1848年，不要改成 20XX 年。

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

篇幅要求：要点不超过 30 条，术语表不超过 15 条。必须完整输出全部 5 个小节，
不要在最后一句中途停下（前面写太多导致「术语表」「待确认」写不完）。

只提炼原文里真实出现的内容，不得补充原文没有的信息：
- 原文说"总书记"就写"总书记"，不要自行补全成具体人名；反之亦然
- 不要添加原文没有的引语、出处、数字或结论
- 原文含糊的地方写进「待确认」，不要靠常识补全
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


def _normalize_base(base: str) -> str:
    """把用户填的地址归一化成 OpenAI 兼容的 base URL（自动补 /v1）。"""
    b = (base or "").strip().rstrip("/")
    if not b:
        return ""
    # 有人直接把 /chat/completions 整段贴进来，截掉后缀
    for suffix in ("/chat/completions", "/completions"):
        if b.endswith(suffix):
            b = b[: -len(suffix)].rstrip("/")
    if not b.endswith("/v1") and "/v1" not in b:
        b += "/v1"
    return b


def _parse_model_names(data: Any) -> List[str]:
    """从 /models 响应里提取模型名（兼容 data / models / result 等字段）。"""
    items: Any = None
    if isinstance(data, dict):
        for key in ("data", "models", "result", "items"):
            v = data.get(key)
            if isinstance(v, list):
                items = v
                break
    elif isinstance(data, list):
        items = data

    names: List[str] = []
    seen = set()
    for it in items or []:
        name = ""
        if isinstance(it, str):
            name = it.strip()
        elif isinstance(it, dict):
            for key in ("id", "name", "model", "model_name", "slug"):
                v = it.get(key)
                if isinstance(v, str) and v.strip():
                    name = v.strip()
                    break
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def fetch_model_list(base: str, api_key: str = "", timeout: Optional[int] = None) -> List[str]:
    """探测 OpenAI 兼容服务可用的模型列表（GET {base}/models）。

    填了地址和 Key 就能列出候选模型，不用背模型名。
    服务不提供 /models 时抛 RuntimeError（提示手填），不影响其他功能。
    """
    normalized = _normalize_base(base)
    if not normalized:
        raise RuntimeError("请先填写接口地址 Base URL")

    raw = (base or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)].rstrip("/")
    urls = [normalized + "/models"]
    alt = raw + "/models"
    if alt and alt not in urls:
        urls.append(alt)

    headers = {"Accept": "application/json", "User-Agent": "stt-workbench"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    tmo = timeout or config.LLM_API_TIMEOUT

    last = ""
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=tmo) as resp:
                data = json.loads(resp.read().decode("utf-8", "ignore"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "ignore")[:200]
            except Exception:
                pass
            if exc.code in (401, 403):
                # 不回显响应体：个别服务会把请求内容原样吐回来
                raise RuntimeError(f"鉴权失败（HTTP {exc.code}）：API Key 无效或无权限") from exc
            if exc.code == 404:
                last = "HTTP 404"
                continue
            last = f"HTTP {exc.code}: {body}"
            continue
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            continue

        names = _parse_model_names(data)
        if names:
            return names
        last = "响应里没有模型列表"

    raise RuntimeError(
        f"拉不到模型列表（{last}）。该服务可能不提供 /models 接口，请手动填写模型名。"
    )


class ContentBlocked(RuntimeError):
    """内容审核拦截（服务端返回 new_sensitive 等）。

    这类错误重试多少次都一样，必须让上层直接跳过受影响的片段，
    否则会白等好几轮退避，还可能把整批内容丢掉。
    """


def resolve_probe_target(backend: str, override: Optional[Dict[str, str]] = None) -> tuple[str, str]:
    """解析探测模型列表用的 (base, api_key)。

    与 _resolve_online 同源，多一条兜底：地址和已保存配置一致时，
    即使后端名不同也复用已保存的 Key，省得用户重复输入。
    """
    base, _model, key = _resolve_online(backend, override=override)
    if not key:
        user = load_user_config()
        saved_base = (user.get("base") or "").strip().rstrip("/")
        if saved_base and saved_base == base.strip().rstrip("/") and user.get("api_key"):
            key = user["api_key"]
    return base, key


def _openai_client(base: str, api_key: str = "", model: str = "local-model"):
    """任意 OpenAI 兼容服务（MiniMax / DeepSeek / 通义 / Kimi / 本地 server）。

    只用标准库 urllib 实现，不引入额外依赖；带指数退避重试。
    """
    base_url = _normalize_base(base)
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
                choice = (data.get("choices") or [{}])[0]
                # 推理型模型（如 MiniMax-M1/M2/M3）正文可能落在 reasoning_content
                text = (msg.get("content") or "").strip()
                if not text:
                    text = (msg.get("reasoning_content") or "").strip()
                text = _strip_think(text)
                # 推理模型可能把 max_tokens 全花在思考上，正文一个字都没有——
                # 这种情况下必须让上层知道（否则会被当成"模型没改"）。
                finish = choice.get("finish_reason")
                if finish == "length" and not text:
                    usage = data.get("usage") or {}
                    log.warning(
                        "响应被 max_tokens 截断（正文为空）：思考用了 %s token，"
                        "上限 %s。条目要拆得更小。",
                        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                        usage.get("completion_tokens"))
                    last = "响应被 max_tokens 截断（思考耗尽预算）"
                elif text:
                    if finish == "length":
                        log.warning("响应被 max_tokens 截断，正文可能不完整")
                    return text
                else:
                    last = "空响应"
            except urllib.error.HTTPError as exc:
                # 带上响应体便于排查（原来只有状态码，看不出原因）
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "ignore")[:300]
                except Exception:
                    pass
                last = f"HTTP {exc.code}: {detail}"
                if "new_sensitive" in detail or "sensitive" in detail.lower():
                    # 内容审核拦截：重试多少次都一样，直接放弃这一段
                    log.warning("内容审核拦截，跳过本段: %s", detail[:160])
                    raise ContentBlocked(detail[:160]) from exc
                if exc.code in (422, 429, 503):
                    # 限流：多等一会儿再试，别把退避压到几秒钟
                    time.sleep(min(20 * attempt, 60))
                else:
                    time.sleep(min(2 ** attempt, 10))
                log.warning("LLM 调用失败（第 %d 次）: %s", attempt, last)
                continue
            except Exception as exc:  # 网络抖动/限流 → 退避重试
                last = f"{type(exc).__name__}: {exc}"
                log.warning("LLM 调用失败（第 %d 次）: %s", attempt, last[:200])
            if attempt < retries:
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"LLM 调用失败：{last}")

    return generate


def test_connection(backend: Optional[str] = None,
                   override: Optional[Dict[str, str]] = None,
                   llm=None) -> Dict[str, Any]:
    """连通性自检：发一个极小的请求，确认后端与 Key 可用。

    override 传入界面尚未保存的 base/model/api_key，可直接试连。
    llm 传入已存在的 LocalLLM（本地后端），避免再加载一份 8B 权重。
    """
    name = (backend or load_user_config().get("backend") or config.LLM_BACKEND or "local").lower()
    if name in ("local", "mlx", "mlx-lm"):
        # 本地后端要真的跑一次才作数：服务进程持有 GPU 时是通的。
        # 优先复用调用方传进来的实例，别再单独加载一份 8B 权重。
        try:
            client = llm if llm is not None else LocalLLM(backend="local")
            out = client.complete("回复两个字：好的", "只回复「好的」两个字。",
                                  max_tokens=16, temperature=0.1)
            return {"ok": True, "backend": "local", "info": client.info,
                    "message": f"本地模型可用，回应：{(out or '')[:40]}"}
        except Exception as exc:
            return {"ok": False, "backend": "local",
                    "message": f"本地模型不可用：{type(exc).__name__}: {exc}"}
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

    @property
    def budget_extra(self) -> int:
        """推理模型（M2/M3 系列）的思考预算余量；本地模型不需要。"""
        name = (self._backend or load_user_config().get("backend")
                or config.LLM_BACKEND or "local").strip().lower()
        if name in ("local", "mlx", "mlx-lm"):
            return 0
        return config.LLM_API_MAX_TOKENS

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

# 整批润色失败时的拆半上限：连续这么多次整批没结果就放弃本组，
# 拆半的深度上限。这两个值是防止「服务一挂就指数级放大请求量」的保护。
MAX_EMPTY_BATCHES = 4
MAX_SPLIT_DEPTH = 2


def polish_segments(
    llm: LocalLLM,
    segments: Sequence[Dict],
    context_head: str = "",
    batch_size: int = 16,
    max_chars: int = 2200,
    align_threshold: float = 0.45,
) -> List[str]:
    """对原始文本批量润色，返回与 segments 等长的改写文本列表。

    三道保险：
    1. 输出用带序号的 JSON 行，模型漏行也不会让后续条目整体错位；
    2. 每条结果做与原文的字符相似度校验，明显不对应的退回原文；
    3. 整批失败时的拆半重试有**次数与深度上限**——否则 16 条一批失败会
       指数级放大请求量（每级 2 个提示词 × 3 次重试），服务一挂就是十几个
       小时出不来。超过上限就整批退回原文，让流程往前走。
    """
    out: List[str] = [""] * len(segments)
    batch: List[int] = []
    buf_len = 0
    empty_batches = 0        # 连续「整批没结果」的次数
    aborted = False          # 熔断后本组剩下的段直接退回原文

    def _fallback(i: int) -> None:
        src = (segments[i].get("raw") or segments[i].get("text") or "").strip()
        orig = (segments[i].get("text") or "").strip()
        out[i] = _safe_base(src, orig)

    def flush(idxs: List[int], depth: int = 0) -> None:
        nonlocal empty_batches, aborted
        if not idxs:
            return
        if aborted:
            for i in idxs:
                _fallback(i)
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
        budget = getattr(llm, "budget_extra", 0)
        for system in (POLISH_SYSTEM, POLISH_SYSTEM_FALLBACK):
            try:
                raw_out = llm.complete(prompt, system,
                                       max_tokens=max_tokens_for(idxs, extra=budget),
                                       temperature=0.25)
            except Exception as exc:
                log.warning("润色批次失败: %s", exc)
                raw_out = ""
            raw_out = _strip_think(raw_out)
            if raw_out.strip():
                break
        mapping = _parse_indexed(raw_out, len(idxs))
        if not mapping and len(idxs) > 1:
            # 整批都没拿到结果（模型空回复 / 被 max_tokens 截断）：
            # 拆成两半重来，小批量更容易成功，避免一批 16 段全退回原文。
            empty_batches += 1
            if empty_batches > MAX_EMPTY_BATCHES or depth >= MAX_SPLIT_DEPTH:
                log.warning(
                    "润色连续 %d 批无结果（深度 %d），本组剩余 %d 段退回原文",
                    empty_batches, depth, len(idxs))
                aborted = True
                for i in idxs:
                    _fallback(i)
                return
            log.warning("润色批次 %d 条无结果，拆半重试", len(idxs))
            mid = len(idxs) // 2
            flush(idxs[:mid], depth + 1)
            flush(idxs[mid:], depth + 1)
            return
        empty_batches = 0
        # 模型只写出了一部分条目：把没写到的拆小重试，别让它们白白退回规整稿。
        # 零散段落最容易触发这个——模型要为每段单独推理，思考 token 会吃满
        # 预算导致响应被截断（实测 12 条一批 reasoning_tokens 就到了 20220）。
        handled = set(range(len(idxs)))
        missing = [pos for pos in range(len(idxs)) if pos not in mapping]
        if missing and len(missing) < len(idxs) and depth < 2:
            log.info("润色批次只拿到 %d/%d 条，剩下 %d 条拆小重试",
                     len(mapping), len(idxs), len(missing))
            handled = set(range(len(idxs))) - set(missing)
            sub = [idxs[p] for p in missing]
            # 一次最多 4 条：条目越少模型想得越少，响应越不容易被截断
            for k in range(0, len(sub), 4):
                flush(sub[k:k + 4], depth + 1)
        for pos, i in enumerate(idxs):
            if pos not in handled:
                continue
            src = (segments[i].get("raw") or segments[i].get("text") or "").strip()
            # 原始转写稿（未经规整），数字保真的第二基准。
            # 规整阶段若把年份压坏一位（例如 二零二零年 → 二零年），
            # 候选结果即使过了 clean 基准的校验，也会被这里拦下来。
            orig = (segments[i].get("text") or "").strip()
            # 模型回填的是提示词里给的编号（段落在本批中的绝对位置），
            # 这里按位置取；若模型自行从 1 开始重新编号，再退回批内序号兜底。
            # 早前只按批内序号取，第 2 批往后整批对不上号，模型的改写全被丢弃。
            cand = (mapping.get(i) or mapping.get(pos) or "").strip()
            if not cand:
                out[i] = _safe_base(src, orig)
                continue
            if not _aligned(src, cand, align_threshold):
                log.info("润色结果与原文不匹配（第 %d 条），保留原文", pos + 1)
                out[i] = _safe_base(src, orig)
                continue
            if not _numbers_preserved(src, cand) or not _digits_intact(src, cand) or (
                    orig and orig != src and not _numbers_preserved(orig, cand)):
                log.info("润色结果丢失原文数字（第 %d 条），保留原文", pos + 1)
                out[i] = _safe_base(src, orig)
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


def _safe_base(src: str, orig: str) -> str:
    """润色不可用时的回退基准。

    优先用规整后的文本；只有当规整文本把原文数字弄丢了，才退回原始
    转写稿——宁可句子不够顺，也不能把年份/金额写错。
    """
    if orig and (not _numbers_preserved(orig, src) or not _digits_intact(orig, src)):
        return orig
    return src


def _aligned(src: str, cand: str, threshold: float = 0.30) -> bool:
    """校验润色结果确实对应原文（字符相似度过低说明条目错位或模型跑偏）。"""
    if not src or not cand:
        return False
    if src == cand:
        return True
    import difflib

    return difflib.SequenceMatcher(None, src, cand).ratio() >= threshold


# 四位中文数字年份（如 二零二零 / 二零二二 / 二〇二五）
# 语音识别偶尔会写成 5 位（多一个「二」：二二零二三年），值取最后 4 位。
_CN_YEAR_RE = re.compile(r"[零〇一二三四五六七八九]{4,5}(?=年)")
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
# 现代年份的值域。语音识别把年份写坏时（「二五二五年」其实是 2025 年）
# 错值落在这个区间之外——这种值不能拿来卡模型，否则模型改对了反被判丢年份。
PLAUSIBLE_YEAR_RANGE = (1900, 2100)


def _cn_year_to_int(s: str) -> Optional[int]:
    """把「二零二零」这样的中文数字年份转成 2020。"""
    try:
        return int("".join(str(_CN_DIGITS[c]) for c in s))
    except (KeyError, ValueError):
        return None


def _years_of(text: str) -> set:
    """文本里的年份数值：中文数字年份 + 阿拉伯数字年份。"""
    out = set()
    for m in _CN_YEAR_RE.finditer(text or ""):
        v = _cn_year_to_int(m.group(0)[-4:])
        if v is not None:
            out.add(v)
    # 任何 4 位阿拉伯数字都算年份候选（含 1679/1847 这类历史年份）
    for m in re.finditer(r"(?<!\d)\d{4}(?!\d)", text or ""):
        out.add(int(m.group(0)))
    return out


def _year_repair(v: int) -> Optional[int]:
    """写坏的年份可能的还原值：「二五二五年」→ 2525 → 2025。

    语音识别把「二零二五」写成「二五二五」时，逐位读是 2525，但按
    「20 + 后两位」还原才是 2025。这里只给出候选，不做决定——
    真年份（1679/1847/1848）也得靠模型结合上下文判断，程序不替它猜。
    """
    tail = v % 100
    cand = 2000 + tail
    lo, hi = PLAUSIBLE_YEAR_RANGE
    if lo <= cand <= hi:
        return cand
    return None


def _numbers_preserved(src: str, cand: str) -> bool:
    """校验润色结果没有丢数字。

    模型偶尔会把四位中文数字年份读成三位（「二零二零年」→「二零年」），
    年份是事实性信息，丢了就是硬伤，所以这里做硬校验：原文里的每个年份
    （按数值比对，中文数字与阿拉伯数字等价）在改写后必须仍然存在。

    校验分两种情况：
    - 合理年份（1900~2100，如 2025/2026）：必须原样存活；
    - 不合理的年份（语音识别写坏的，如「二五二五年」逐位读是 2525）：
      只要模型把它还原成合理的年份就算过，否则模型改对了反而会被判成
      丢年份、整段退回原文——这正是高太山/王峰那几处年份修不掉的根因。
    """
    lo, hi = PLAUSIBLE_YEAR_RANGE
    src_all = _years_of(src)
    cand_years = _years_of(cand)
    for v in src_all:
        if lo <= v <= hi:
            # 合理年份：必须还在（中文写法或阿拉伯写法都算）
            if v not in cand_years:
                return False
            continue
        # 不合理年份：原样留着（可能是 1679 这种真历史年份），
        # 或者被还原成合理年份，都算过
        if v in cand_years:
            continue
        rep = _year_repair(v)
        if rep is not None and rep in cand_years:
            continue
        return False
    return True


# 真正的数字才算数：三位以上的数字串、跟单位相邻的两位数字串、阿拉伯数字。
# 「一一定」「统一一下」「大杀四四方」这种含叠字的普通词不计入，
# 否则模型正常顺句（把口吃去掉）会被误判成丢数字。
_CN_NUM_CHARS = "零〇一二三四五六七八九幺"
_NUM_UNIT_CHARS = "年月日元万亿号届季％%倍秒厘后前初"
RE_NUM_TOKEN = re.compile(
    rf"[{_CN_NUM_CHARS}]{{3,}}"
    rf"|[{_CN_NUM_CHARS}]{{2}}(?=[{_NUM_UNIT_CHARS}{_CN_NUM_CHARS}])"
    rf"|\d+"
)


def _digit_weight(text: str) -> int:
    """文本里"数字信息"的重量（数字字符个数）。

    模型把「二零二零」写成「2020」重量不变（4→4），把「二零二零年」写成
    「二零年」重量减半（4→2），所以可以据此判断有没有丢数字。

    年份串统一按 4 位算：语音识别偶尔多写一个字（「二二零二三年」），
    那是写坏了而不是多了一个信息，按 5 位算会把正确的改写（2023年）误判成丢数字。
    """
    text = _CN_YEAR_RE.sub(lambda m: "〇" * 4, text or "")
    return sum(len(m.group(0)) for m in RE_NUM_TOKEN.finditer(text))


def _digits_intact(src: str, cand: str) -> bool:
    """数字信息没变少：中文数字改成阿拉伯数字不算丢，只丢字符才算。"""
    return _digit_weight(cand) >= _digit_weight(src)


def max_tokens_for(idxs: Sequence[int], extra: int = 0) -> int:
    """单次调用的 token 上限。

    extra 是给推理模型（M2/M3 系列）预留的思考预算：它们的思考会先吃掉几千
    token，上限给小了会 finish_reason=length 把正文一起砍没。
    """
    return max(1024, min(32768, len(idxs) * 320 + extra))


def _iter_json_objects(text: str):
    """从一段文本里逐个抠出 JSON 对象（容忍前后夹杂其它文字、单个对象截断）。

    推理模型常在 JSON 行后面接思考文字，或把对象截断成 {"i":0,"t":"半个
    ；按「整行必须是合法 JSON」来解析会把整批结果都丢掉。
    """
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    yield json.loads(text[start:i + 1])
                except Exception:
                    pass
                start = -1


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
    # 推理模型经常在 JSON 后面接一段思考文字，或把 JSON 截断，
    # 所以这里按「行内找 {…}」来提取，而不是要求整行正好是 JSON。
    for line in text.splitlines():
        line = line.strip()
        if "{" not in line:
            continue
        for obj in _iter_json_objects(line):
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

    # 不再按行/空行硬切兜底：模型的思考文字长度常常刚好等于预期条数，
    # 硬切会把无关内容当成改写结果写进稿子。宁可这一批全部退回原文。
    # 编号异常时按顺序回填
    if result and sorted(result) != list(range(expect)):
        ordered = [result[k] for k in sorted(result)]
        result = {i: v for i, v in enumerate(ordered)}
    return result


# ---------------------------------------------------------------- 知识库提炼

CHUNK_SYSTEM = """你是一名中文文稿编辑。你会收到一段演讲稿的中间部分，请提取其中值得保留的信息。

输出要点，每行一条，不要标题、不要解释：
- （事实、数据、观点、案例、结论；保留具体数字与人名地名）"""

KB_SECTIONS = ("## 摘要", "## 要点", "## 术语表", "## 待确认")


def _kb_complete(text: str) -> bool:
    """知识库是否 5 个小节齐全（防止 max_tokens 截断后残缺）。"""
    return all(sec in (text or "") for sec in KB_SECTIONS)


# 常见的政治/人物专名：知识库里出现但原文没有 = 模型自行补全，属于加戏
_NAME_TOKENS = ("习近平", "毛泽东", "邓小平", "江泽民", "胡锦涛", "李克强",
                "周恩来", "刘少奇", "陈云", "朱德", "马克思", "恩格斯",
                "列宁", "斯大林", "孙中山", "蒋介石")


def _kb_adds_names(kb: str, source: str) -> bool:
    """检查知识库有没有自行补出原文没有的人物专名。

    原文说"总书记"而知识库写"习近平指出"，就是模型加了原文没有的信息，
    违反"只做提炼不加戏"的要求，要退回重试。

    注意要按集合整体比对：原文写"马克思在《共产党宣言》里提出"，
    知识库写成"马克思和恩格斯创立了…"，多出来的"恩格斯"同样属于加戏。
    """
    kb_names = {n for n in _NAME_TOKENS if n in kb}
    if not kb_names:
        return False
    src_names = {n for n in _NAME_TOKENS if n in source}
    return bool(kb_names - src_names)


_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def _strip_think(text: str) -> str:
    """去掉推理模型塞在正文里的 <think>…</think> 思考块。

    MiniMax M 系列（推理模型）会把思考过程和正文一起放进 content，
    不剥掉的话：JSON 行解析会被思考里的编号/引号带偏，
    极端情况下整个思维链会被当成润色结果写进稿子。
    """
    if not text:
        return ""
    out = _THINK_RE.sub("", text)
    low = out.lower()
    i = low.find("<think>")
    if i >= 0:          # 未闭合的思考块：从 <think> 起到结尾全部丢掉
        out = out[:i]
    return out.strip()


def _extract_part(llm: LocalLLM, part: str, label: str, budget: int) -> str:
    """提炼一段文字。内容审核拦截时切细重试，尽量把内容救回来。"""
    try:
        return llm.complete(
            f"这是演讲稿的{label}：\n\n{part}",
            CHUNK_SYSTEM,
            max_tokens=1400 + budget,
            temperature=0.2,
        )
    except ContentBlocked:
        # 整段触发审核：切成两半各试一次，能救一半算一半
        mid = len(part) // 2
        got = []
        for sub, sub_label in ((part[:mid], f"{label}上半"), (part[mid:], f"{label}下半")):
            if not sub.strip():
                continue
            try:
                out = llm.complete(
                    f"这是演讲稿的{sub_label}：\n\n{sub}",
                    CHUNK_SYSTEM,
                    max_tokens=1400 + budget,
                    temperature=0.2,
                )
            except ContentBlocked:
                continue
            except Exception:
                continue
            if out.strip():
                got.append(out.strip())
        if got:
            return "\n".join(got)
        raise
    except Exception as exc:
        log.warning("分段提炼失败 (%s): %s", label, exc)
        return ""


# 「要点提炼」的长度上限。最终汇总提示词 = 整理稿前 6000 字 + merged，
# 而内网引擎的上下文只有 32768 token（中文 1 字≈1 token）：merged 一旦
# 超过两万多字就会 context_length_exceeded，整篇知识库退化成空壳。
KB_MERGE_LIMIT = 9000
# 合并要点时单次请求的切片大小——保证「合并」这一步自己也不会撞上下文上限。
KB_MERGE_SLICE = 6000


def _compress_notes(llm: LocalLLM, merged: str, budget: int) -> str:
    """把各段要点合并去重。

    merged 可能几万字（长稿 20 段 × 每段 2000 字）。原来一次性塞进提示词，
    「合并」这一步自己就先撞上引擎的上下文上限——异常被吞掉后 merged 原样
    返回，最终汇总再炸一次，整篇知识库只剩空壳。所以这里按 KB_MERGE_SLICE
    切片，各自压缩后再拼接；某一批失败就退回该批原文，不丢内容。
    """
    slices = [merged[i:i + KB_MERGE_SLICE] for i in range(0, len(merged), KB_MERGE_SLICE)]

    def _one(idx: int) -> str:
        chunk = slices[idx]
        try:
            out = llm.complete(
                f"以下是演讲稿各部分的要点（第 {idx + 1}/{len(slices)} 批），"
                f"请去重合并：\n\n{chunk}",
                CHUNK_SYSTEM,
                max_tokens=2600 + budget,
                temperature=0.2,
            )
            return out.strip() if out and out.strip() else chunk
        except Exception as exc:
            log.warning("要点合并第 %d/%d 批失败，保留原文: %s",
                        idx + 1, len(slices), exc)
            return chunk

    workers = max(1, min(int(config.LLM_KB_WORKERS), len(slices)))
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            outs = list(pool.map(_one, range(len(slices))))
    else:
        outs = [_one(i) for i in range(len(slices))]
    return "\n".join(x for x in outs if x.strip())


def build_knowledge_base(llm: LocalLLM, full_text: str, title: str = "") -> str:
    """长稿走「分段提炼 → 汇总」两段式，避免一次性超长输入。

    分段提炼默认并发跑：长稿能切出几十段，串行的话慢模型（内网 27B 之类）
    要串行几十分钟到几小时，单段还容易撞上超时。并发之后同样的量只要几分钟。
    """
    if not full_text.strip():
        return ""

    # ---- 第一步：分段提炼（并发）
    size = 2600
    parts = [full_text[i:i + size] for i in range(0, len(full_text), size)]
    notes: List[Optional[str]] = [None] * len(parts)
    budget = getattr(llm, "budget_extra", 0)
    workers = max(1, min(int(config.LLM_KB_WORKERS), len(parts)))

    def _work(idx: int) -> None:
        try:
            out = _extract_part(llm, parts[idx], f"第 {idx + 1}/{len(parts)} 部分", budget)
        except Exception as exc:
            log.warning("分段提炼失败 (%d/%d): %s", idx + 1, len(parts), exc)
            out = ""
        notes[idx] = out

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_work, range(len(parts))))
    else:
        for idx in range(len(parts)):
            _work(idx)

    keep = [n.strip() for n in notes if n and n.strip()]
    if not keep:
        return ""

    merged = "\n".join(keep)
    # 合并后仍过长时再压一轮（切片压，不会自己撞上下文上限）
    if len(merged) > KB_MERGE_LIMIT:
        merged = _compress_notes(llm, merged, budget)
    # 兜底：压不动就截断。整理稿 6000 字 + merged 必须留在引擎上下文里，
    # 否则最终汇总会 context_length_exceeded，知识库只剩 46 字符的空壳。
    if len(merged) > KB_MERGE_LIMIT:
        merged = merged[:KB_MERGE_LIMIT]

    # ---- 第二步：结构化汇总
    title_hint = f"标题：{title}\n\n" if title else ""
    extra = ""
    merged_cap = len(merged)
    out = ""
    for attempt in range(2):
        prompt = (
            f"{title_hint}以下是演讲整理稿及要点提炼，请产出结构化的知识库资料。\n\n"
            f"【整理稿】\n{full_text[:6000]}\n\n"
            f"【要点提炼】\n{merged[:merged_cap]}\n" + extra
        )
        try:
            out = llm.complete(prompt, KB_SYSTEM,
                               max_tokens=max(8192, budget), temperature=0.2)
        except Exception as exc:
            log.warning("知识库生成失败: %s", exc)
            out = ""
            # 多半是上下文装不下（要点太长）→ 砍掉一半要点再试
            merged_cap = merged_cap // 2
            continue
        if out.strip() and _kb_complete(out) and not _kb_adds_names(out, full_text):
            break
        if attempt == 0:
            # 上一次被 max_tokens 截断或漏了小节：重来一次，并明确要求收着写
            log.warning("知识库输出不完整（可能被截断），重试一次")
            extra = (
                "\n\n注意：上一次输出被截断或缺少小节。请务必输出全部 5 个小节"
                "（标题/摘要/要点/术语表/待确认），要点控制在 25 条以内、术语表 10 条以内，"
                "务必以「## 待确认」小节结尾。"
                "另外只写原文里真实出现过的人名与称谓，不要自行补全。"
            )
    out = (out or "").strip()
    # 兜底：仍缺小节就补空节，保证结构完整
    for sec in KB_SECTIONS:
        if sec not in out:
            out += f"\n\n{sec}\n- 无"
    return out
