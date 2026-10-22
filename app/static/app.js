/* 本地语音转文字工作台 —— 前端逻辑 */

const $ = (id) => document.getElementById(id);

const state = {
  docId: null,
  doc: null,
  files: [],
  peaks: null,       // Int16Array, 每桶 min/max
  peaksPerSec: 20,
  selected: new Set(),
  pollTimer: null,
  pollingJob: false,
  lastActive: -1,
  dragStart: null,   // 拖拽选段起点
  llmCfg: {},        // 界面里保存的在线模型配置
};

/* ------------------------------------------------------------------ 工具 */
const api = async (url, opts = {}) => {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  return res.headers.get('content-type')?.includes('json') ? res.json() : res.text();
};

function fmtTs(sec, ms = false) {
  sec = Math.max(0, sec || 0);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  const ss = ms ? s.toFixed(3).padStart(6, '0') : String(Math.floor(s)).padStart(2, '0');
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${ss}`;
}

function fmtSize(bytes) {
  if (!bytes) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0, n = bytes;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
}

let toastTimer = null;
function toast(msg, ms = 2600) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add('hidden'), ms);
}

function debounce(fn, wait) {
  let t = null;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), wait); };
}

/* ------------------------------------------------------------------ 启动 */
async function init() {
  const health = await api('/api/health');
  $('envInfo').textContent = `模型 ${health.asr_model} · 语言 ${health.language} · 润色 ${health.llm_model}`;
  $('sourceDir').textContent = health.source_dir;
  $('sourceDir').title = health.source_dir;

  // 润色后端下拉：本地默认，其余为可选的网络 API
  await refreshLlmBackends();

  $('audio').addEventListener('timeupdate', onTimeUpdate);
  $('audio').addEventListener('loadedmetadata', () => {
    $('timeDur').textContent = fmtTs($('audio').duration);
  });
  $('audio').addEventListener('play', () => { $('btnPlay').textContent = '❚❚'; });
  $('audio').addEventListener('pause', () => { $('btnPlay').textContent = '▶'; });

  $('btnPlay').onclick = togglePlay;
  $('speedSel').onchange = (e) => { $('audio').playbackRate = parseFloat(e.target.value); };
  $('btnFollow').onclick = (e) => {
    const on = e.target.classList.toggle('active');
    e.target.textContent = on ? '跟随' : '不跟随';
  };
  $('btnTranscribe').onclick = startTranscribe;
  $('btnRules').onclick = runRules;
  $('btnPolishAll').onclick = () => polish(null);
  $('btnPolishSel').onclick = () => polish([...state.selected]);
  $('btnKB').onclick = buildKB;
  $('btnLlmSet').onclick = openLlmPanel;
  $('btnLlmClose').onclick = closeLlmPanel;
  $('btnLlmSave').onclick = saveLlmConfig;
  $('btnLlmTest').onclick = testLlmBackend;
  $('btnLlmClear').onclick = clearLlmConfig;
  $('cfgBackend').onchange = onCfgBackendChange;
  $('llmPanel').onclick = (e) => { if (e.target === $('llmPanel')) closeLlmPanel(); };
  $('btnKbCopy').onclick = async () => {
    try { await navigator.clipboard.writeText($('kbText').textContent); toast('已复制到剪贴板'); }
    catch (e) { toast('复制失败，请手动选择'); }
  };
  $('btnKbToggle').onclick = () => {
    const box = $('kbBox');
    const collapsed = box.classList.toggle('collapsed');
    $('btnKbToggle').textContent = collapsed ? '展开' : '收起';
  };

  $('btnExport').onclick = (e) => {
    e.stopPropagation();
    $('exportMenu').parentElement.classList.toggle('open');
  };
  document.addEventListener('click', () => $('exportMenu').parentElement.classList.remove('open'));
  $('exportMenu').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b || !state.docId) return;
    window.location = `/api/doc/${encodeURIComponent(state.docId)}/export?fmt=${b.dataset.fmt}&mode=${b.dataset.mode}`;
  });

  ['showRaw', 'showPolish'].forEach(id => $(id).addEventListener('change', renderSegments));

  initWave();
  await loadFiles();
  // 只有一个文件时自动选中，省一次点击
  if (!state.docId && state.files.length === 1) {
    await selectFile(state.files[0].id);
  }
  window.addEventListener('resize', drawWave);
}

/* ------------------------------------------- 在线模型 API 设置（⚙️ 模型设置） */
const LLM_LOCAL = 'local';
const LLM_CUSTOM = 'custom';

function setLlmMsg(text, bad = false) {
  const el = $('llmMsg');
  if (!el) return;
  el.textContent = text || '';
  el.style.color = bad ? '#ef5350' : '';
}

/** 刷新顶部「润色后端」下拉：本地默认，未配 Key 的在线后端置灰。 */
async function refreshLlmBackends() {
  try {
    const bl = await api('/api/llm/backends');
    const sel = $('llmSel');
    if (!sel) return;
    const prev = sel.value;
    const ready = {};
    sel.innerHTML = '';
    for (const b of bl.backends) {
      ready[b.id] = !!b.ready;
      const opt = document.createElement('option');
      opt.value = b.id;
      opt.textContent = b.id === LLM_LOCAL ? b.label : b.label + (b.ready ? '' : '（未配置 Key）');
      opt.disabled = !b.ready;
      sel.appendChild(opt);
    }
    const pick = [prev, bl.default, LLM_LOCAL].find(id => id && ready[id]);
    sel.value = pick || LLM_LOCAL;
    sel.onchange = () => toast(`润色后端：${sel.options[sel.selectedIndex].textContent}`);
  } catch (e) { /* 后端列表拿不到就用默认本地 */ }
}

async function openLlmPanel() {
  $('llmPanel').classList.remove('hidden');
  setLlmMsg('');
  await loadLlmConfig();
}

function closeLlmPanel() {
  $('llmPanel').classList.add('hidden');
  $('cfgKey').value = '';   // 不在内存里留明文 Key
}

/** 读取已保存配置并回填表单（Key 只显示末 4 位，不回填明文）。 */
async function loadLlmConfig() {
  let cfg = {};
  try {
    cfg = await api('/api/llm/config');
  } catch (e) {
    setLlmMsg('读取配置失败：' + e.message, true);
    return;
  }
  state.llmCfg = cfg;
  const presets = cfg.presets || {};
  const sel = $('cfgBackend');
  sel.innerHTML = '';
  const addOpt = (v, t) => {
    const o = document.createElement('option');
    o.value = v;
    o.textContent = t;
    sel.appendChild(o);
  };
  addOpt(LLM_LOCAL, '本地模型（离线，默认）');
  for (const [id, p] of Object.entries(presets)) addOpt(id, p.label || id);
  addOpt(LLM_CUSTOM, '自定义 OpenAI 兼容（任意 Base URL）');

  const cur = cfg.backend || LLM_LOCAL;
  sel.value = [...sel.options].some(o => o.value === cur) ? cur : LLM_LOCAL;

  $('cfgKey').value = '';
  $('cfgKey').placeholder = cfg.has_key
    ? `已保存（${cfg.key_hint}），留空表示不修改`
    : 'sk-…（留空表示用环境变量）';

  const p = presets[cur] || {};
  $('cfgBase').value = cur === LLM_LOCAL ? '' : (cfg.base || p.base || '');
  $('cfgModel').value = cur === LLM_LOCAL ? '' : (cfg.model || p.model || '');

  if (cfg.saved) {
    setLlmMsg(`当前配置：${cur}${cfg.has_key ? ' · Key ' + cfg.key_hint : ''}`);
  }
}

/** 切换后端时自动带出默认地址/模型。 */
function onCfgBackendChange() {
  const id = $('cfgBackend').value;
  const presets = (state.llmCfg || {}).presets || {};
  if (id === LLM_LOCAL) {
    $('cfgBase').value = '';
    $('cfgModel').value = '';
    setLlmMsg('已选本地模型：离线运行，不需要 Key');
    return;
  }
  if (id === LLM_CUSTOM) {
    setLlmMsg('填入 Base URL 与模型名，兼容任何 OpenAI 协议服务');
    return;
  }
  const p = presets[id] || {};
  if (p.base) $('cfgBase').value = p.base;
  if (p.model) $('cfgModel').value = p.model;
  setLlmMsg(p.key_env
    ? `默认从环境变量 ${p.key_env} 读 Key，也可在下方直接填写`
    : '请填写 API Key 后保存');
}

/** 收集表单里的测试/保存参数。 */
function llmFormValues() {
  return {
    backend: $('cfgBackend').value,
    base: $('cfgBase').value.trim(),
    model: $('cfgModel').value.trim(),
    api_key: $('cfgKey').value.trim(),
  };
}

async function saveLlmConfig() {
  const v = llmFormValues();
  if (v.backend === LLM_LOCAL) {
    setLlmMsg('本地模型无需配置，直接用顶部下拉选择「本地」即可', true);
    return;
  }
  if (!v.base) {
    setLlmMsg('请填写接口地址 Base URL', true);
    return;
  }
  setLlmMsg('保存中…');
  try {
    const body = { backend: v.backend, base: v.base, model: v.model };
    if (v.api_key) body.api_key = v.api_key;   // 留空 = 不修改已保存的 Key
    const r = await api('/api/llm/config', { method: 'POST', body: JSON.stringify(body) });
    setLlmMsg(r.message || '已保存');
    $('cfgKey').value = '';
    await loadLlmConfig();
    await refreshLlmBackends();
    toast('模型配置已保存');
  } catch (e) {
    setLlmMsg('保存失败：' + e.message, true);
  }
}

async function testLlmBackend() {
  const v = llmFormValues();
  if (v.backend === LLM_LOCAL) {
    setLlmMsg('本地模型需要 GPU（Metal），无法用网络方式自检', true);
    return;
  }
  setLlmMsg('测试中…');
  try {
    const r = await api('/api/llm/test', { method: 'POST', body: JSON.stringify(v) });
    setLlmMsg(r.message || (r.ok ? '连通正常' : '连接失败'), !r.ok);
  } catch (e) {
    setLlmMsg('测试失败：' + e.message, true);
  }
}

async function clearLlmConfig() {
  setLlmMsg('清除中…');
  try {
    const r = await api('/api/llm/config', { method: 'POST', body: JSON.stringify({ clear: true }) });
    setLlmMsg(r.message || '已清除');
    $('cfgKey').value = '';
    await loadLlmConfig();
    await refreshLlmBackends();
    toast('已清除在线模型配置');
  } catch (e) {
    setLlmMsg('清除失败：' + e.message, true);
  }
}

async function loadFiles() {
  const data = await api('/api/files');
  state.files = data.files;
  const ul = $('fileList');
  ul.innerHTML = '';
  if (!data.files.length) {
    ul.innerHTML = '<li class="muted small" style="padding:10px">目录中没有音频文件</li>';
    return;
  }
  for (const f of data.files) {
    const li = document.createElement('li');
    li.dataset.id = f.id;
    const st = f.doc?.status || 'pending';
    const statusText = { ready: '已完成', running: '转写中', pending: '未转写', error: '出错' }[st] || st;
    li.innerHTML = `
      <div class="fname">${f.name}</div>
      <div class="fmeta">
        <span class="dot ${st}"></span>${statusText}
        <span>·</span><span>${fmtSize(f.size)}</span>
        ${f.doc?.segments ? `<span>·</span><span>${f.doc.segments} 段</span>` : ''}
      </div>`;
    li.onclick = () => selectFile(f.id);
    ul.appendChild(li);
  }
}

function markActiveFile() {
  document.querySelectorAll('#fileList li').forEach(li => {
    li.classList.toggle('active', li.dataset.id === state.docId);
  });
}

/* ------------------------------------------------------------------ 选文件 */
async function selectFile(id) {
  state.docId = id;
  state.selected.clear();
  markActiveFile();
  try {
    const doc = await api(`/api/doc/${encodeURIComponent(id)}`);
    state.doc = doc;
    applyDoc(doc);
  } catch (e) {
    // 还没转写过
    state.doc = null;
    $('nowFile').textContent = state.files.find(f => f.id === id)?.name || id;
    $('nowMeta').textContent = '尚未转写，点击「开始转写」';
    $('segments').innerHTML = '<div class="empty">该文件还没有转写结果，点击「开始转写」开始</div>';
    $('segCount').textContent = '0 段';
    $('audio').removeAttribute('src');
    $('kbBox').classList.add('hidden');
  }
  // 预取波形
  loadPeaks(id);
}

function applyDoc(doc) {
  state.doc = doc;
  $('nowFile').textContent = doc.source || doc.id;
  const m = doc.media || {};
  const segs = doc.segments || [];
  $('nowMeta').textContent = `${fmtTs(m.duration)} · ${(m.sample_rate || 0)}Hz · ${m.channels || 0}ch · ${segs.length} 段`;
  $('segCount').textContent = `${segs.length} 段`;
  $('audio').src = `/api/doc/${encodeURIComponent(doc.id)}/audio`;
  $('audio').load();

  if (doc.knowledge_base) {
    $('kbText').textContent = doc.knowledge_base;
    $('kbBox').classList.remove('hidden');
  } else {
    $('kbBox').classList.add('hidden');
  }
  renderSegments();
  if (doc.status === 'running' || doc.job?.running) startPolling();
}

function renderSegments() {
  const box = $('segments');
  const segs = state.doc?.segments || [];
  if (!segs.length) {
    box.innerHTML = '<div class="empty">暂无内容</div>';
    return;
  }
  const showRaw = $('showRaw').checked;
  const showPol = $('showPolish').checked;
  const frag = document.createDocumentFragment();

  segs.forEach((seg, i) => {
    const div = document.createElement('div');
    div.className = 'seg' + (state.selected.has(i) ? ' sel' : '');
    div.dataset.i = i;

    const ts = document.createElement('div');
    ts.className = 'ts';
    ts.textContent = fmtTs(seg.start);
    ts.title = '点击跳转播放';
    ts.onclick = () => seek(seg.start);

    const body = document.createElement('div');
    body.className = 'body';

    if (showRaw) body.appendChild(makeLine('原文', seg.raw, 'raw', i, 'raw'));
    if (showPol) {
      const hasPolish = !!(seg.polished || seg.edited);
      const label = seg.edited ? '已改' : (hasPolish ? '润色' : '待润色');
      body.appendChild(makeLine(label, seg.edited || seg.polished || '', 'pol', i, 'polished', hasPolish));
    }
    if (!showRaw && !showPol) body.appendChild(makeLine('内容', seg.raw, 'raw', i, 'raw'));

    div.appendChild(ts);
    div.appendChild(body);
    frag.appendChild(div);
  });

  box.innerHTML = '';
  box.appendChild(frag);
  bindSegmentEvents();
  highlightActive(state.lastActive, false);
}

function makeLine(label, text, cls, i, field, hasContent) {
  const line = document.createElement('div');
  line.className = 'line';

  const tag = document.createElement('span');
  tag.className = `tag ${cls}`;
  tag.textContent = label;
  line.appendChild(tag);

  const txt = document.createElement('div');
  txt.className = 'txt' + (cls === 'pol' && hasContent ? ' polished' : '');
  txt.contentEditable = 'true';
  txt.spellcheck = false;
  if (!text) { txt.textContent = '（空）'; txt.classList.add('empty'); }
  else txt.textContent = text;
  txt.dataset.i = i;
  txt.dataset.field = field;
  line.appendChild(txt);

  const acts = document.createElement('div');
  acts.className = 'acts';
  if (cls === 'pol') {
    const b1 = document.createElement('button');
    b1.className = 'mini';
    b1.textContent = 'AI';
    b1.title = '只润色这一段';
    b1.onclick = (e) => { e.stopPropagation(); polish([i]); };
    acts.appendChild(b1);

    const b2 = document.createElement('button');
    b2.className = 'mini';
    b2.textContent = '还原';
    b2.title = '撤销人工修改，回到 AI/识别结果';
    b2.onclick = async (e) => {
      e.stopPropagation();
      await api(`/api/doc/${encodeURIComponent(state.docId)}/segment/${i}/reset`, { method: 'POST' });
      await reloadDoc();
    };
    acts.appendChild(b2);
  }
  line.appendChild(acts);
  return line;
}

function bindSegmentEvents() {
  const autoSave = () => $('autoSave').checked;
  document.querySelectorAll('#segments .txt').forEach(el => {
    el.addEventListener('focus', () => el.classList.remove('empty'));
    el.addEventListener('blur', async () => {
      const i = parseInt(el.dataset.i, 10);
      const field = el.dataset.field;
      const val = el.textContent.trim();
      const seg = state.doc?.segments?.[i];
      if (!seg) return;
      const isRaw = field === 'raw';
      if (isRaw) {
        if ((seg.raw || '') === val) return;
        seg.raw = val;
      } else {
        if (seg.edited === val) return;
        seg.edited = val;
        el.classList.toggle('polished', !!val);
        el.classList.toggle('edited', !!val && seg.edited_manually === true);
      }
      if (autoSave()) {
        try {
          await api(`/api/doc/${encodeURIComponent(state.docId)}/segment/${i}`, {
            method: 'POST',
            body: JSON.stringify({ text: val, field: isRaw ? 'raw' : 'edited' }),
          });
          toast('已保存');
        } catch (e) { toast('保存失败: ' + e.message); }
      }
    });
  });

  // 点击段落 = 选中（可多选），双击 = 跳转播放
  document.querySelectorAll('#segments .seg').forEach(el => {
    const i = parseInt(el.dataset.i, 10);
    el.addEventListener('click', (e) => {
      if (e.target.closest('.txt') || e.target.closest('button')) return;
      if (e.shiftKey && state.dragStart !== null) {
        const [a, b] = [state.dragStart, i].sort((x, y) => x - y);
        for (let k = a; k <= b; k++) state.selected.add(k);
      } else {
        if (state.selected.has(i)) state.selected.delete(i);
        else state.selected.add(i);
        state.dragStart = i;
      }
      renderSegments();
      $('selInfo').textContent = state.selected.size ? `已选 ${state.selected.size} 段` : '';
    });
    el.addEventListener('dblclick', (e) => {
      if (e.target.closest('.txt') || e.target.closest('button')) return;
      seek(state.doc.segments[i].start);
    });
  });
}

/* ------------------------------------------------------------------ 播放 */
function togglePlay() {
  const a = $('audio');
  if (!a.src) return;
  a.paused ? a.play() : a.pause();
}

function seek(sec) {
  const a = $('audio');
  if (!a.src) return;
  a.currentTime = Math.max(0, sec);
  a.play();
}

function onTimeUpdate() {
  const t = $('audio').currentTime;
  $('timeCur').textContent = fmtTs(t);
  const segs = state.doc?.segments || [];
  let idx = -1;
  for (let i = 0; i < segs.length; i++) {
    if (t >= segs[i].start && t < segs[i].end) { idx = i; break; }
    if (t < segs[i].start) break;
  }
  if (idx !== state.lastActive) highlightActive(idx, true);
  drawWave();
}

function highlightActive(idx, scroll) {
  state.lastActive = idx;
  document.querySelectorAll('#segments .seg').forEach(el => {
    el.classList.toggle('active', parseInt(el.dataset.i, 10) === idx);
  });
  if (scroll && idx >= 0 && $('followPlay').checked) {
    const el = document.querySelector(`#segments .seg[data-i="${idx}"]`);
    if (el) el.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }
}

/* ------------------------------------------------------------------ 波形 */
function initWave() {
  const cv = $('wave');
  cv.addEventListener('click', (e) => {
    const r = cv.getBoundingClientRect();
    const ratio = (e.clientX - r.left) / r.width;
    const dur = $('audio').duration || 0;
    if (dur) seek(ratio * dur);
  });
}

async function loadPeaks(docId) {
  try {
    const res = await fetch(`/api/doc/${encodeURIComponent(docId)}/peaks`);
    if (!res.ok) throw new Error('no peaks');
    const buf = await res.arrayBuffer();
    state.peaks = new Int16Array(buf);
    drawWave();
  } catch (e) {
    state.peaks = null;
  }
}

function drawWave() {
  const cv = $('wave');
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0a0d12';
  ctx.fillRect(0, 0, w, h);

  const dur = $('audio').duration || (state.doc?.media?.duration || 0);
  const cur = $('audio').currentTime || 0;

  // 段落分界线
  const segs = state.doc?.segments || [];
  if (dur > 0) {
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (const s of segs) {
      const x = Math.round((s.start / dur) * w) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
  }

  const peaks = state.peaks;
  if (!peaks || !dur) {
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.font = '12px sans-serif';
    ctx.fillText('波形将在转写完成后生成', 12, h / 2);
    return;
  }

  const nBuckets = peaks.length / 2;
  const bucketsPerSec = nBuckets / dur;
  const px = w / nBuckets;
  const mid = h / 2;

  for (let i = 0; i < nBuckets; i++) {
    const lo = peaks[i * 2] / 32768;
    const hi = peaks[i * 2 + 1] / 32768;
    const t = i / bucketsPerSec;
    const x = i * px;
    const played = t <= cur;
    ctx.fillStyle = played ? '#4f9dff' : 'rgba(154,165,181,0.5)';
    const top = mid - hi * mid * 0.92;
    const bot = mid - lo * mid * 0.92;
    ctx.fillRect(x, top, Math.max(1, px), Math.max(1, bot - top));
  }

  // 播放头
  if (dur > 0) {
    const x = (cur / dur) * w;
    ctx.fillStyle = '#ff5f56';
    ctx.fillRect(x - 1, 0, 2, h);
  }
}

/* ------------------------------------------------------------------ 动作 */
async function startTranscribe() {
  if (!state.docId) { toast('请先选择文件'); return; }
  const btn = $('btnTranscribe');
  btn.disabled = true;
  try {
    await api('/api/transcribe', {
      method: 'POST',
      body: JSON.stringify({
        id: state.docId,
        engine: $('engineSel').value,
        align: true,
        rules: true,
      }),
    });
    toast('已提交转写任务');
    startPolling();
  } catch (e) {
    toast('提交失败: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

function startPolling() {
  if (state.pollTimer) return;
  state.pollTimer = setInterval(pollStatus, 1200);
}

function stopPolling() {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

async function pollStatus() {
  if (!state.docId) { stopPolling(); return; }
  try {
    const doc = await api(`/api/doc/${encodeURIComponent(state.docId)}`);
    const p = doc.progress || {};
    const job = doc.job;
    const pp = doc.polish_progress;
    const wrap = $('progressWrap');
    const busy = doc.status === 'running' || (job && job.running);

    if (busy) {
      wrap.classList.remove('hidden');
      let pct = 0, text = '';
      if (pp && pp.total) {
        pct = Math.round((pp.done / pp.total) * 100);
        text = `本地 LLM 润色中　${pp.done}/${pp.total} 段　${pct}%`;
      } else if (doc.status === 'running') {
        pct = p.total ? Math.round((p.done / p.total) * 100) : 2;
        const eta = p.eta ? `　预计剩余 ${Math.round(p.eta)} 秒` : '';
        text = `${p.stage || '处理中'}　${p.done || 0}/${p.total || 0} 段${p.message ? '　' + p.message : ''}${eta}`;
      } else {
        text = job?.message || '处理中…';
      }
      $('progressFill').style.width = pct + '%';
      $('progressText').textContent = text;
    } else {
      wrap.classList.add('hidden');
    }

    // 只有任务真正结束才收尾：润色/知识库不改变 doc.status，
    // 只看 job.running，否则会在任务还在跑的时候就误报「处理完成」并停止轮询。
    if (state.pollingJob && !busy) {
      state.pollingJob = false;
      stopPolling();
      if (job && job.stage === 'error') {
        toast('任务失败: ' + (job.error || '未知错误'));
      } else {
        await reloadDoc();
        toast('处理完成');
      }
      await loadFiles();
      return;
    }
    if (doc.segments && doc.segments.length !== (state.doc?.segments?.length || 0)) {
      state.doc = doc;
      renderSegments();
    } else if (state.doc) {
      state.doc = doc;
    }
    $('segCount').textContent = `${(doc.segments || []).length} 段`;
  } catch (e) { /* 忽略瞬时错误 */ }
}

async function reloadDoc() {
  const doc = await api(`/api/doc/${encodeURIComponent(state.docId)}`);
  state.doc = doc;
  renderSegments();
  $('kbText').textContent = doc.knowledge_base || '';
  $('kbBox').classList.toggle('hidden', !doc.knowledge_base);
}

async function runRules() {
  if (!state.docId) return;
  toast('正在做规则清理…');
  try {
    await api(`/api/doc/${encodeURIComponent(state.docId)}/rules`, {
      method: 'POST', body: JSON.stringify({ level: 'light' }),
    });
    await reloadDoc();
    toast('规则清理完成');
  } catch (e) { toast('失败: ' + e.message); }
}

async function polish(indices) {
  if (!state.docId) return;
  const backend = ($('llmSel') && $('llmSel').value) || '';
  const label = indices ? `已提交润色 ${indices.length} 段` : '已提交全篇润色';
  toast(label + '…');
  try {
    await api(`/api/doc/${encodeURIComponent(state.docId)}/polish`, {
      method: 'POST', body: JSON.stringify({ indices, level: 'standard', backend }),
    });
    state.pollingJob = true;
    startPolling();
    toast('润色任务进行中，可在顶部看进度');
  } catch (e) { toast('失败: ' + e.message); }
}

async function buildKB() {
  if (!state.docId) return;
  const backend = ($('llmSel') && $('llmSel').value) || '';
  toast('已提交知识库提炼任务…');
  try {
    await api(`/api/doc/${encodeURIComponent(state.docId)}/knowledge_base`, {
      method: 'POST', body: JSON.stringify({ backend }),
    });
    state.pollingJob = true;
    startPolling();
    toast('知识库提炼中，完成后自动显示');
  } catch (e) { toast('失败: ' + e.message); }
}

init().catch(e => {
  document.body.insertAdjacentHTML('afterbegin',
    `<div style="padding:20px;color:#ef5350">初始化失败：${e.message}</div>`);
});
