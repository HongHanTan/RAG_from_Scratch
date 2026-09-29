/* The retrieval inspector page: plain JavaScript, no framework, no build step.
 *
 * The one structural rule here: /api/ask and /api/compare are two independent
 * requests. Comparison re-runs every strategy and calls the LLM once per
 * translating strategy, so it takes seconds; the other four panels come from
 * the single fast /api/ask run. Awaiting both before painting anything would
 * make every panel as slow as the slowest, so each request clears its own
 * panels' loading state when it resolves, and each panel shows its own error.
 */

'use strict';

var ASK_PANELS = [
  'chunk-table', 'translation-trace', 'stage-timings', 'vector-projection'
];
var COMPARE_PANELS = ['strategy-comparison'];

/* --- small DOM helpers ---------------------------------------------------- */

function el(tag, className, text) {
  var node = document.createElement(tag);
  if (className) { node.className = className; }
  if (text !== undefined && text !== null) { node.textContent = String(text); }
  return node;
}

function panel(id) {
  return document.getElementById(id);
}

function part(id, attribute) {
  return panel(id).querySelector('[' + attribute + ']');
}

function bodyOf(id) {
  return part(id, 'data-body');
}

function startPanels(ids) {
  ids.forEach(function (id) {
    part(id, 'data-loading').hidden = false;
    var error = part(id, 'data-error');
    error.hidden = true;
    error.textContent = '';
  });
}

function finishPanels(ids) {
  ids.forEach(function (id) { part(id, 'data-loading').hidden = true; });
}

function failPanels(ids, message) {
  ids.forEach(function (id) {
    part(id, 'data-loading').hidden = true;
    var error = part(id, 'data-error');
    error.textContent = message;
    error.hidden = false;
  });
}

/* A non-200 carries FastAPI's `detail`; a dropped connection carries nothing.
 * Both have to reach the panel as text -- a spinner left spinning forever is
 * the one failure mode this page must not have. Each request site calls
 * `fetch` itself and pipes the response through here, so there is no single
 * shared request that could accidentally couple the fast panels to the slow
 * one. */
function asJSON(response) {
  if (!response.ok) {
    return response.json().then(
      function (payload) { throw new Error(payload.detail || response.status); },
      function () { throw new Error('HTTP ' + response.status); }
    );
  }
  return response.json();
}

function fixed(value, places) {
  return (typeof value === 'number' && isFinite(value))
    ? value.toFixed(places) : '-';
}

function excerpt(text, limit) {
  text = (text || '').replace(/\s+/g, ' ').trim();
  return text.length > limit ? text.slice(0, limit) + '…' : text;
}

/* --- metadata ------------------------------------------------------------- */

function loadMeta() {
  var line = document.getElementById('index-meta');
  fetch('/api/meta').then(asJSON).then(function (meta) {
    line.textContent = meta.chunks + ' chunks, ' + meta.dim + ' dimensions, '
      + 'index ' + meta.index_mode
      + (meta.embedding_model ? ', embeddings ' + meta.embedding_model : '')
      + (meta.llm_model ? ', llm ' + meta.llm_model : '');
    var select = document.getElementById('strategy');
    var names = meta.strategies || [];
    if (names.length) {
      select.textContent = '';
      names.forEach(function (name) {
        var option = el('option', null, name);
        option.value = name;
        select.appendChild(option);
      });
    }
  }).catch(function (error) {
    line.textContent = 'Could not read index metadata: ' + error.message;
  });
}

/* --- panel 1: the chunk table --------------------------------------------- */

function renderChunks(trace) {
  var body = bodyOf('chunk-table');
  body.textContent = '';
  var rows = trace.retrieved || [];
  if (!rows.length) {
    body.appendChild(el('p', 'empty', 'Nothing was retrieved for this question.'));
    return;
  }
  var table = el('table', 'grid');
  var head = el('tr');
  ['rank', 'score', 'doc', 'chunk', 'text'].forEach(function (name) {
    head.appendChild(el('th', null, name));
  });
  table.appendChild(head);

  rows.forEach(function (item) {
    var chunk = item.chunk || {};
    var row = el('tr');
    row.appendChild(el('td', 'num', item.rank));

    /* The kind travels with the number. A maxsim score of 12 and a cosine
     * score of 0.5 are not one scale, and a column headed only "score" would
     * read a reranked run as a tenfold jump in quality. */
    var score = el('td', 'score');
    score.appendChild(el('span', 'value', fixed(item.score, 4)));
    score.appendChild(el('span', 'kind', item.score_kind || 'unknown'));
    row.appendChild(score);

    row.appendChild(el('td', 'doc', chunk.doc_id || '-'));
    row.appendChild(el('td', 'num', chunk.index));
    row.appendChild(el('td', 'text', excerpt(chunk.text, 280)));
    table.appendChild(row);
  });
  body.appendChild(table);

  var notes = trace.notes || [];
  if (notes.length) {
    var list = el('ul', 'notes');
    notes.forEach(function (note) { list.appendChild(el('li', null, note)); });
    body.appendChild(list);
  }
}

/* --- panel 2: strategy comparison ----------------------------------------- */

function renderComparison(payload) {
  var body = bodyOf('strategy-comparison');
  body.textContent = '';
  var results = payload.results || [];
  if (!results.length) {
    body.appendChild(el('p', 'empty', 'No strategies were compared.'));
    return;
  }
  var table = el('table', 'grid');
  var head = el('tr');
  ['strategy', 'queries', 'chunks', 'top score', 'total ms', 'top chunk']
    .forEach(function (name) { head.appendChild(el('th', null, name)); });
  table.appendChild(head);

  results.forEach(function (result) {
    var trace = result.trace || {};
    var retrieved = trace.retrieved || [];
    var top = retrieved[0];
    var row = el('tr');
    row.appendChild(el('td', 'doc', result.strategy));
    row.appendChild(el('td', 'num', (trace.queries || []).length));
    row.appendChild(el('td', 'num', retrieved.length));

    var score = el('td', 'score');
    if (top) {
      score.appendChild(el('span', 'value', fixed(top.score, 4)));
      score.appendChild(el('span', 'kind', top.score_kind || 'unknown'));
    } else {
      score.textContent = '-';
    }
    row.appendChild(score);

    row.appendChild(el('td', 'num', fixed(trace.total_ms, 1)));
    row.appendChild(el('td', 'text',
      top ? excerpt((top.chunk || {}).chunk_id, 60) : '-'));
    table.appendChild(row);
  });
  body.appendChild(table);
  body.appendChild(el('p', 'hint',
    'Scores are only comparable between rows that share a score kind.'));
}

/* --- panel 3: the translation trace --------------------------------------- */

function renderTranslation(trace) {
  var body = bodyOf('translation-trace');
  body.textContent = '';

  var steps = trace.translation || [];
  if (!steps.length) {
    body.appendChild(el('p', 'empty',
      'No translation: "' + (trace.strategy || 'direct')
      + '" searched the question as written.'));
  } else {
    var list = el('ol', 'translation');
    steps.forEach(function (step) {
      var item = el('li');
      item.appendChild(el('span', 'kind', step.kind));
      item.appendChild(el('span', 'step-text', step.text));
      list.appendChild(item);
    });
    body.appendChild(list);
  }

  var queries = trace.queries || [];
  if (queries.length) {
    body.appendChild(el('h3', null,
      'Queries actually searched (' + queries.length + ')'));
    var searched = el('ul', 'queries');
    queries.forEach(function (query) {
      searched.appendChild(el('li', null, query));
    });
    body.appendChild(searched);
  }
}

/* --- panel 4: stage timings ----------------------------------------------- */

function renderTimings(trace) {
  var body = bodyOf('stage-timings');
  body.textContent = '';
  var timings = trace.timings || [];
  if (!timings.length) {
    body.appendChild(el('p', 'empty', 'No stages were timed.'));
    return;
  }
  var longest = timings.reduce(function (most, timing) {
    return Math.max(most, timing.ms || 0);
  }, 0) || 1;

  var chart = el('div', 'timings');
  timings.forEach(function (timing) {
    var row = el('div', 'timing');
    row.style.marginLeft = ((timing.depth || 0) * 18) + 'px';
    row.appendChild(el('span', 'stage-name', timing.name));
    var track = el('span', 'bar-track');
    var bar = el('span', 'bar');
    bar.style.width = Math.max(1, (timing.ms / longest) * 100) + '%';
    track.appendChild(bar);
    row.appendChild(track);
    row.appendChild(el('span', 'stage-ms', fixed(timing.ms, 1) + ' ms'));
    chart.appendChild(row);
  });
  body.appendChild(chart);
  body.appendChild(el('p', 'hint',
    'Total (top-level stages only): ' + fixed(trace.total_ms, 1) + ' ms'));
}

/* --- panel 5: the projection ---------------------------------------------- */

function renderProjection(trace) {
  var projection = trace.projection || {};
  var points = projection.chunks || [];
  var query = projection.query || null;
  var note = document.getElementById('projection-note');
  note.textContent = projection.note
    || 'a projection of the embeddings, not the space itself';

  var canvas = document.getElementById('projection-canvas');
  var height = 340;
  var width = Math.max(320, canvas.parentNode.clientWidth || 640);
  var ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.width = width + 'px';
  canvas.style.height = height + 'px';

  var ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  var all = points.slice();
  if (query) { all.push(query); }
  if (!all.length) {
    ctx.fillStyle = '#6b7280';
    ctx.font = '13px sans-serif';
    ctx.fillText('Nothing to project.', 14, 24);
    return;
  }

  var xs = all.map(function (p) { return p[0]; });
  var ys = all.map(function (p) { return p[1]; });
  var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
  var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
  var spanX = (maxX - minX) || 1;
  var spanY = (maxY - minY) || 1;
  var pad = 34;

  function sx(x) { return pad + ((x - minX) / spanX) * (width - 2 * pad); }
  function sy(y) { return height - pad - ((y - minY) / spanY) * (height - 2 * pad); }

  ctx.strokeStyle = '#e5e7eb';
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

  points.forEach(function (point, i) {
    var x = sx(point[0]), y = sy(point[1]);
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#2563eb';
    ctx.fill();
    ctx.fillStyle = '#111827';
    ctx.font = '11px sans-serif';
    ctx.fillText(String(i + 1), x + 9, y - 7);
  });

  /* The query is a red cross, not another dot: it is a different kind of
   * thing from the chunks and must not read as one of them. */
  if (query) {
    var qx = sx(query[0]), qy = sy(query[1]);
    var arm = 9;
    ctx.strokeStyle = '#dc2626';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(qx - arm, qy - arm);
    ctx.lineTo(qx + arm, qy + arm);
    ctx.moveTo(qx - arm, qy + arm);
    ctx.lineTo(qx + arm, qy - arm);
    ctx.stroke();
    ctx.fillStyle = '#dc2626';
    ctx.font = 'bold 11px sans-serif';
    ctx.fillText('query', qx + arm + 3, qy + 4);
  }
}

/* --- the two independent requests ----------------------------------------- */

function runAsk(params) {
  startPanels(ASK_PANELS);
  fetch('/api/ask?' + params).then(asJSON).then(function (trace) {
    renderChunks(trace);
    renderTranslation(trace);
    renderTimings(trace);
    renderProjection(trace);
    finishPanels(ASK_PANELS);
  }).catch(function (error) {
    failPanels(ASK_PANELS, 'Request failed: ' + error.message);
  });
}

function runCompare(params) {
  startPanels(COMPARE_PANELS);
  fetch('/api/compare?' + params).then(asJSON).then(function (payload) {
    renderComparison(payload);
    finishPanels(COMPARE_PANELS);
  }).catch(function (error) {
    failPanels(COMPARE_PANELS, 'Request failed: ' + error.message);
  });
}

function submit(event) {
  event.preventDefault();
  var question = document.getElementById('question').value;
  if (!question.trim()) { return; }

  var shared = new URLSearchParams();
  shared.set('q', question);
  var k = document.getElementById('k').value;
  if (k) { shared.set('k', k); }
  if (document.getElementById('rerank').checked) { shared.set('rerank', 'true'); }

  var askParams = new URLSearchParams(shared);
  askParams.set('strategy', document.getElementById('strategy').value);

  // Fired back to back and never awaited together: the four fast panels paint
  // as soon as /api/ask lands, while comparison is still running.
  runAsk(askParams.toString());
  runCompare(shared.toString());
}

document.getElementById('query-form').addEventListener('submit', submit);
loadMeta();
