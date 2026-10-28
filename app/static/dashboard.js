/* 状态仪表盘：轮询 /api/status，渲染作业进度 + 硬件占用 */
(function () {
  'use strict';

  var STAGE_LABEL = {
    queued: '排队中',
    probing: '读取音频信息',
    converting: '转码 16kHz',
    segmenting: '静音检测与分段',
    loading_model: '加载模型',
    recognizing: '识别中',
    cleaning: '口语清理',
    preview: '生成预览与波形',
    polishing: 'AI 润色',
    knowledge_base: '提炼知识库',
    done: '完成',
    error: '出错',
    paused: '已暂停',
    cancelled: '已取消',
  };
  var KIND_LABEL = {
    asr: '转写',
    polish: '润色',
    kb: '知识库',
    rules: '规则清理',
    unknown: '任务',
  };
  var STATE_LABEL = {
    running: '运行中',
    paused: '已暂停',
    stale: '可能卡住',
    died: '进程已退出',
    done: '完成',
    error: '出错',
  };

  function $(id) { return document.getElementById(id); }

  function fmtDur(s) {
    if (s === null || s === undefined) return '—';
    s = Math.max(0, Math.round(s));
    var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    if (h) return h + ' 小时 ' + m + ' 分';
    if (m) return m + ' 分 ' + sec + ' 秒';
    return sec + ' 秒';
  }

  function num(v, digits) {
    if (v === null || v === undefined) return '—';
    if (typeof v !== 'number') return String(v);
    return digits === undefined ? String(v) : v.toFixed(digits);
  }

  function esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&').replace(/</g, '<').replace(/>/g, '>')
      .replace(/"/g, '"');
  }

  function bar(pct, cls) {
    var p = (pct === null || pct === undefined) ? 0 : Math.max(0, Math.min(100, pct));
    return '<div class="job-bar"><i style="width:' + p + '%"></i></div>';
  }

  function jobCard(j) {
    var kind = KIND_LABEL[j.kind] || j.kind;
    var stage = STAGE_LABEL[j.stage] || j.stage;
    var pct = j.pct === null || j.pct === undefined ? null : j.pct;
    var docId = j.doc_id || j.id || '';
    var meta = [];
    if (j.done !== null && j.total) {
      meta.push('<span>进度 <b>' + j.done + ' / ' + j.total + '</b></span>');
    }
    if (j.elapsed_s) meta.push('<span>已跑 <b>' + fmtDur(j.elapsed_s) + '</b></span>');
    if (j.eta_s) meta.push('<span>预计剩余 <b>' + fmtDur(j.eta_s) + '</b></span>');
    if (j.pid) meta.push('<span>PID <b>' + j.pid + '</b></span>');
    if (j.cpu_pct !== undefined && j.cpu_pct !== null) {
      meta.push('<span>进程 CPU <b>' + num(j.cpu_pct, 1) + '%</b></span>');
    }
    if (j.rss_mb !== undefined && j.rss_mb !== null) {
      meta.push('<span>内存 <b>' + num(j.rss_mb, 1) + ' MB</b></span>');
    }
    if (j.message) meta.push('<span>' + esc(j.message) + '</span>');
    if (j.error) meta.push('<span style="color:var(--err)">' + esc(j.error) + '</span>');

    // 这项作业用的是哪个模型
    var modelTag = '';
    if (j.model) {
      modelTag = '<span class="job-model" title="后端：' + esc(j.backend || '—') + '">'
        + '模型 ' + esc(j.model) + '</span>';
    }

    // 暂停 / 恢复 / 取消
    var actions = '';
    if (j.state === 'paused') {
      actions = '<button class="job-btn primary" data-job="' + esc(docId) +
        '" data-action="resume">恢复</button>' +
        '<button class="job-btn danger" data-job="' + esc(docId) +
        '" data-action="cancel">取消</button>';
    } else if (j.state === 'running' || j.state === 'stale') {
      actions = '<button class="job-btn" data-job="' + esc(docId) +
        '" data-action="pause">暂停</button>' +
        '<button class="job-btn danger" data-job="' + esc(docId) +
        '" data-action="cancel">取消</button>';
    }

    return '<div class="job kind-' + esc(j.kind) + ' state-' + esc(j.state) + '">' +
      '<div class="job-top">' +
        '<span class="job-kind">' + esc(kind) + '</span>' +
        '<span class="job-title">' + esc(j.id) + '</span>' +
        modelTag +
        '<span class="job-stage">' + esc(stage) +
          (j.state !== 'running' && j.state !== 'stale' && j.state !== 'paused'
            && STATE_LABEL[j.state] ? ' · ' + STATE_LABEL[j.state] : '') +
        '</span>' +
        '<span class="job-pct">' + (pct === null ? '' : pct + '%') + '</span>' +
      '</div>' +
      bar(pct) +
      '<div class="job-meta">' + meta.join('') + '</div>' +
      (actions ? '<div class="job-actions">' + actions + '</div>' : '') +
    '</div>';
  }

  function meter(name, label, pct, value, extraCls) {
    var cls = 'meter ' + (extraCls || '');
    if (pct === null || pct === undefined) cls += ' unknown';
    var w = (pct === null || pct === undefined) ? 0 : Math.max(0, Math.min(100, pct));
    return '<div class="' + cls + '">' +
      '<div class="meter-top"><span>' + label + '</span><span class="val">' + value + '</span></div>' +
      '<div class="meter-bar"><i style="width:' + w + '%"></i></div>' +
    '</div>';
  }

  function renderSystem(s) {
    s = s || {};
    var out = '';
    out += meter('cpu', 'CPU', s.cpu_pct,
      (s.cpu_pct === null || s.cpu_pct === undefined)
        ? '—'
        : num(s.cpu_pct, 1) + '%' + (s.cpu_cores ? ' / ' + s.cpu_cores + ' 核' : ''));
    out += meter('mem', '内存', s.mem_pct,
      (s.mem_used_gb === null || s.mem_used_gb === undefined)
        ? '—'
        : num(s.mem_used_gb, 1) + ' / ' + num(s.mem_total_gb, 1) + ' GB');
    out += meter('gpu', 'GPU', s.gpu_pct,
      (s.gpu_pct === null || s.gpu_pct === undefined)
        ? '—'
        : num(s.gpu_pct, 0) + '%'
          + (s.gpu_mem_gb ? '（' + num(s.gpu_mem_gb, 1) + ' GB）' : ''));
    var low = (s.disk_free_gb !== null && s.disk_free_gb !== undefined && s.disk_free_gb < 20);
    out += meter('disk', '磁盘', s.disk_pct,
      (s.disk_free_gb === null || s.disk_free_gb === undefined)
        ? '—'
        : '剩 ' + num(s.disk_free_gb, 1) + ' GB'
          + (s.disk_total_gb ? ' / ' + num(s.disk_total_gb, 1) + ' GB' : ''),
      'disk' + (low ? ' low' : ''));
    $('sysList').innerHTML = out;
  }

  function renderJobs(jobs) {
    var running = [], done = [];
    (jobs || []).forEach(function (j) {
      if (j.state === 'running' || j.state === 'stale' || j.state === 'died'
          || j.state === 'paused') running.push(j);
      else done.push(j);
    });
    $('jobList').innerHTML = running.length
      ? running.map(jobCard).join('')
      : '<div class="empty-note">当前没有正在运行的任务</div>';

    if (!done.length) {
      $('doneList').innerHTML = '<div class="empty-note">暂无已完成记录</div>';
      return;
    }
    $('doneList').innerHTML = done.slice(0, 12).map(function (j) {
      var mark = j.state === 'error' ? '<span class="bad">✕</span>' : '<span class="ok">✓</span>';
      var dur = j.finished_at && j.started_at
        ? fmtDur(j.finished_at - j.started_at) : '—';
      return '<div class="done-row">' + mark +
        '<span class="who">' + esc(j.id) + '</span>' +
        '<span>' + esc(KIND_LABEL[j.kind] || j.kind) + '</span>' +
        '<span>' + esc(STAGE_LABEL[j.stage] || j.stage) + '</span>' +
        '<span class="dim model">' + esc(j.model || '—') + '</span>' +
        '<span class="dim">' + dur + '</span>' +
        '<span class="dim">' + (j.finished_at
          ? new Date(j.finished_at * 1000).toLocaleString('zh-CN', { hour12: false })
          : '') + '</span>' +
      '</div>';
    }).join('');
  }

  function renderProcs(data) {
    var rows = [];
    (data.jobs || []).forEach(function (j) {
      rows.push('<tr>' +
        '<td>' + esc(j.pid) + '</td>' +
        '<td>' + esc(KIND_LABEL[j.kind] || j.kind) + '</td>' +
        '<td>' + esc(j.id) + '</td>' +
        '<td class="model">' + esc(j.model || '—') + '</td>' +
        '<td>' + (j.cpu_pct === null || j.cpu_pct === undefined ? '—' : num(j.cpu_pct, 1) + '%') + '</td>' +
        '<td>' + (j.rss_mb === null || j.rss_mb === undefined ? '—' : num(j.rss_mb, 1) + ' MB') + '</td>' +
        '<td class="' + (j.state === 'running' ? '' : 'dim') + '">'
          + (STATE_LABEL[j.state] || j.state) + '</td>' +
      '</tr>');
    });
    var srv = (data.server || {});
    rows.push('<tr>' +
      '<td>' + esc(srv.pid) + '</td>' +
      '<td>服务</td>' +
      '<td class="dim">' + esc(srv.build || '') + '</td>' +
      '<td class="dim">—</td>' +
      '<td>—</td><td>—</td>' +
      '<td class="dim">运行中</td>' +
    '</tr>');
    $('procBody').innerHTML = rows.join('');
  }

  function render(data) {
    $('dashSub').textContent = '构建 ' + (data.server && data.server.build || '—')
      + '　·　输出目录 ' + (data.server && data.server.output_dir || '');
    renderJobs(data.jobs);
    renderSystem(data.system);
    renderProcs(data);
    $('dashFresh').textContent = '更新于 ' + new Date().toLocaleTimeString('zh-CN', { hour12: false });
  }

  var timer = null;

  function setJobMsg(text, bad) {
    var el = $('jobMsg');
    if (!el) return;
    if (!text) { el.hidden = true; el.textContent = ''; return; }
    el.hidden = false;
    el.textContent = text;
    el.className = 'job-msg' + (bad ? ' bad' : '');
  }

  function onJobAction(e) {
    var btn = (e.target && e.target.closest) ? e.target.closest('.job-btn') : null;
    if (!btn) return;
    var docId = btn.getAttribute('data-job') || '';
    var action = btn.getAttribute('data-action') || '';
    if (!docId || !action) return;
    var label = { pause: '暂停', resume: '恢复', cancel: '取消' }[action] || action;
    if (action === 'cancel'
        && !window.confirm('确定取消「' + docId + '」吗？当前批次结束后停止。')) return;
    btn.disabled = true;
    setJobMsg('正在' + label + '「' + docId + '」…');
    fetch('/api/jobs/' + encodeURIComponent(docId) + '/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: action }),
    }).then(function (r) { return r.json(); })
      .then(function (d) {
        setJobMsg((d && d.message) ? d.message : ('已' + label));
        btn.disabled = false;
        tick();
      })
      .catch(function (err) {
        setJobMsg('操作失败：' + err.message, true);
        btn.disabled = false;
      });
  }

  function tick() {
    fetch('/api/status')
      .then(function (r) { return r.json(); })
      .then(render)
      .catch(function (e) {
        $('dashFresh').textContent = '接口出错：' + e.message;
      });
  }

  function start() {
    if (timer) return;
    tick();
    timer = setInterval(tick, 2000);
  }

  document.addEventListener('DOMContentLoaded', function () {
    $('btnRefresh').addEventListener('click', tick);
    $('jobList').addEventListener('click', onJobAction);
    start();
  });
})();
