// Physiome-ODE leaderboard. Data comes from data/results.json (built by website/build_leaderboard.py)
// and samples/<id>.json (built by website/generate_samples.py); every statistic is computed here.
(function () {
  'use strict';
  // Hero figure: a smooth endocrine oscillator that shows the forecasting task at a glance.
  var HERO = { id: 'LEN01', instance: 3, seed: 41 };
  var EXPLORER_DEFAULT = 'LI01';
  // Colors come from each model's "slot" in website/models.json: 1-8 are the validated categorical
  // palette, 0 is a neutral tone (used for the Elo reference model).
  var STATUS = { paper: ['Paper', 'badge'], verified: ['Verified', 'badge ver'], 'self-reported': ['Self-reported', 'badge self'] };

  function getJSON(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error(url + ': HTTP ' + r.status);
      return r.json();
    });
  }

  getJSON('data/results.json').then(init, function (err) {
    var msg = location.protocol === 'file:'
      ? 'Open this page through a web server: run "python -m http.server" in the docs folder and visit http://localhost:8000.'
      : 'The leaderboard data could not be loaded (' + err.message + ').';
    document.getElementById('lb-body').innerHTML = '<tr><td colspan="8" class="empty"></td></tr>';
    document.querySelector('#lb-body .empty').textContent = msg;
  });

  function init(RAW) {
    var INFO = {};
    RAW.models.forEach(function (m, i) {
      INFO[m.name] = { slot: m.slot || 0, family: m.family, ref: m.ref, code: m.code, status: m.status, source: m.source };
    });
    var MODELS = RAW.models.map(function (m) { return m.name; });
    var K = MODELS.length;
    var ANCHOR = Math.max(0, MODELS.indexOf(RAW.anchor));
    var D = RAW.datasets;
    var N_ALL = D.length;
    var MSE = D.map(function (d) { return MODELS.map(function (m) { return d.res[m][0]; }); });
    var SD = D.map(function (d) { return MODELS.map(function (m) { return d.res[m][1]; }); });
    var BEST = MSE.map(function (v) { return Math.min.apply(null, v); });
    var RANK = MSE.map(function (v) { return v.map(function (x) { return 1 + v.filter(function (y) { return y < x; }).length; }); });

    var byJgd = D.map(function (_, i) { return i; }).sort(function (a, b) { return D[b].jgd - D[a].jgd; });
    var TIER = new Array(N_ALL);
    var third = Math.round(N_ALL / 3);
    byJgd.forEach(function (d, r) { TIER[d] = r < third ? 'high' : r < 2 * third ? 'mid' : 'low'; });

    var state = { tier: 'all', domain: 'all', ch: 'all', view: 'mse', lbSort: { key: 'elo', dir: -1 }, dsSort: { key: 'jgd', dir: -1 } };
    var DEFAULT_DIR = { elo: -1, rank: 1, wins: -1, imp: 1 };

    function $(id) { return document.getElementById(id); }
    function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
    function info(i) { return INFO[MODELS[i]] || { slot: 0, family: '—', ref: '', code: '#', status: 'self-reported' }; }
    function col(i) { var s = info(i).slot; return s ? 'var(--s' + s + ')' : 'var(--muted)'; }
    function fmtJ(x) { return x >= 1 ? x.toFixed(2) : x >= 0.1 ? x.toFixed(3) : x.toPrecision(2); }
    function fmtInt(x) { return Math.round(x).toLocaleString('en-US'); }
    function fmtRatio(r) { return '×' + (r < 10 ? r.toFixed(2) : r < 100 ? r.toFixed(1) : r.toFixed(0)); }
    function prettyDomain(s) { var t = s.replace(/_/g, ' '); return t.charAt(0).toUpperCase() + t.slice(1); }
    function range(n) { var a = []; for (var i = 0; i < n; i++) a.push(i); return a; }
    function mean(a) { return a.reduce(function (s, x) { return s + x; }, 0) / a.length; }

    function mulberry32(a) {
      return function () {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        var t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
      };
    }

    // Bradley–Terry ratings via MM iterations; one virtual tie per pair keeps ratings finite.
    function btElo(idx) {
      var W = new Float64Array(K), N = range(K).map(function () { return new Float64Array(K); });
      var i, j;
      for (i = 0; i < K; i++) for (j = i + 1; j < K; j++) { N[i][j] = N[j][i] = 1; W[i] += 0.5; W[j] += 0.5; }
      idx.forEach(function (d) {
        var v = MSE[d];
        for (var a = 0; a < K; a++) for (var b = a + 1; b < K; b++) {
          N[a][b]++; N[b][a]++;
          if (v[a] < v[b]) W[a]++; else if (v[b] < v[a]) W[b]++; else { W[a] += 0.5; W[b] += 0.5; }
        }
      });
      var p = new Float64Array(K).fill(1);
      for (var it = 0; it < 500; it++) {
        var q = new Float64Array(K), lg = 0, delta = 0;
        for (i = 0; i < K; i++) {
          var den = 0;
          for (j = 0; j < K; j++) if (j !== i) den += N[i][j] / (p[i] + p[j]);
          q[i] = W[i] / den; lg += Math.log(q[i]);
        }
        lg = Math.exp(lg / K);
        for (i = 0; i < K; i++) { q[i] /= lg; delta = Math.max(delta, Math.abs(q[i] - p[i])); }
        p = q;
        if (delta < 1e-10) break;
      }
      return Array.prototype.map.call(p, function (x) { return 1000 + 400 * Math.log10(x / p[ANCHOR]); });
    }

    function computeStats(idx) {
      var n = idx.length;
      var elo = btElo(idx);
      var seed = idx.reduce(function (h, d) { return (Math.imul(h, 31) + d + 1) | 0; }, 2025);
      var rng = mulberry32(seed), B = 300, boot = range(K).map(function () { return []; });
      for (var b = 0; b < B; b++) {
        var s = idx.map(function () { return idx[Math.floor(rng() * n)]; });
        btElo(s).forEach(function (v, i) { boot[i].push(v); });
      }
      var models = range(K).map(function (i) {
        var sorted = boot[i].slice().sort(function (a, c) { return a - c; });
        return {
          i: i,
          elo: elo[i],
          lo: sorted[Math.floor(0.025 * (B - 1))],
          hi: sorted[Math.ceil(0.975 * (B - 1))],
          rank: mean(idx.map(function (d) { return RANK[d][i]; })),
          wins: idx.filter(function (d) { return MSE[d][i] === BEST[d]; }).length,
          imp: 100 * mean(idx.map(function (d) { return (MSE[d][i] - BEST[d]) / MSE[d][i]; }))
        };
      });
      var wr = range(K).map(function (i) {
        return range(K).map(function (j) {
          if (i === j) return null;
          var w = 0;
          idx.forEach(function (d) { var a = MSE[d][i], c = MSE[d][j]; w += a < c ? 1 : a === c ? 0.5 : 0; });
          return { rate: w / n, wins: idx.filter(function (d) { return MSE[d][i] < MSE[d][j]; }).length };
        });
      });
      return { n: n, models: models, wr: wr };
    }

    function sliceIdx() {
      return range(N_ALL).filter(function (i) {
        var d = D[i];
        if (state.tier !== 'all' && TIER[i] !== state.tier) return false;
        if (state.domain !== 'all' && d.domain !== state.domain) return false;
        if (state.ch === 'lo' && d.channels > 5) return false;
        if (state.ch === 'hi' && d.channels <= 5) return false;
        return true;
      });
    }

    // ---------- Leaderboard ----------
    function renderLeaderboard(st) {
      var body = $('lb-body');
      if (!st.n) {
        body.innerHTML = '<tr><td colspan="8" class="empty">No datasets match this slice. Widen a filter or reset.</td></tr>';
        return;
      }
      var byElo = st.models.slice().sort(function (a, b) { return b.elo - a.elo; });
      var pos = {}; byElo.forEach(function (m, r) { pos[m.i] = r + 1; });
      var key = state.lbSort.key, dir = state.lbSort.dir;
      var rows = st.models.slice().sort(function (a, b) { return dir * (a[key] - b[key]) || (b.elo - a.elo); });
      body.innerHTML = rows.map(function (m) {
        var inf = info(m.i);
        return '<tr>' +
          '<td class="c-pos">' + pos[m.i] + '</td>' +
          '<td><div class="mcell"><span class="sw" style="background:' + col(m.i) + '"></span><div><a class="mname" href="' + esc(inf.code) + '">' + esc(MODELS[m.i]) + '</a><span class="mref">' + esc(inf.ref) + '</span></div></div></td>' +
          '<td><span class="chip">' + esc(inf.family) + '</span></td>' +
          '<td><div class="elocell"><span class="elo-num">' + fmtInt(m.elo) + '</span><span class="elo-ci">' + (m.i === ANCHOR ? 'anchor' : '+' + fmtInt(m.hi - m.elo) + ' /−' + fmtInt(m.elo - m.lo)) + '</span></div></td>' +
          '<td class="num">' + m.rank.toFixed(2) + '</td>' +
          '<td class="num">' + m.wins + '<span class="of"> / ' + st.n + '</span></td>' +
          '<td class="num">' + m.imp.toFixed(1) + '%</td>' +
          '<td><span class="' + (STATUS[inf.status] || STATUS['self-reported'])[1] + '"' + (inf.source ? ' title="Results: ' + esc(inf.source) + '"' : '') + '>' + (STATUS[inf.status] || STATUS['self-reported'])[0] + '</span></td>' +
          '</tr>';
      }).join('');
      document.querySelectorAll('.lb th[data-key]').forEach(function (th) {
        if (th.dataset.key === key) th.setAttribute('aria-sort', dir < 0 ? 'descending' : 'ascending');
        else th.removeAttribute('aria-sort');
      });
    }

    // ---------- Elo bar chart (TabArena-style, horizontal) ----------
    var lastStats = null;
    function renderEloChart(st) {
      var box = $('elo-chart');
      if (!st || !st.n) { box.innerHTML = '<p class="empty">No datasets in this slice.</p>'; return; }
      var rows = st.models.slice().sort(function (a, b) { return b.elo - a.elo; });
      var W = Math.max(300, Math.round(box.clientWidth || 800));
      eloWidth = W;
      var narrow = W < 560;
      var LW = narrow ? 116 : 176, RW = narrow ? 46 : 64, rowH = narrow ? 30 : 38, T = 4, AX = 30, barH = narrow ? 14 : 18;
      var H = T + rows.length * rowH + AX;
      var lo = Math.min.apply(null, rows.map(function (m) { return Math.min(m.lo, m.elo); }));
      var hi = Math.max.apply(null, rows.map(function (m) { return Math.max(m.hi, m.elo); }));
      var span = Math.max(hi - lo, 100);
      // Widest-spaced ticks that still give about one label per 90px of plot width.
      var maxTicks = Math.max(3, Math.floor((W - LW - RW) / 90));
      var step = [50, 100, 200, 400, 800, 1600].filter(function (st) { return Math.ceil(span / st) + 1 <= maxTicks; })[0] || 1600;
      var floor = Math.floor((lo - span * 0.1) / 100) * 100, ceil = Math.ceil((hi + span * 0.05) / 100) * 100;
      function x(v) { return LW + (v - floor) / (ceil - floor) * (W - LW - RW); }
      var base = T + rows.length * rowH;
      var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" width="' + W + '" height="' + H + '" role="img" aria-label="Elo rating per model with 95% confidence intervals">';
      for (var v = Math.ceil(floor / step) * step; v <= ceil + 0.1; v += step) {
        s += '<line class="e-grid" x1="' + x(v) + '" x2="' + x(v) + '" y1="' + T + '" y2="' + (base + 4) + '"/>';
        s += '<text class="e-tick" x="' + x(v) + '" y="' + (base + 20) + '" text-anchor="middle">' + fmtInt(v) + '</text>';
      }
      rows.forEach(function (m, r) {
        var cy = T + r * rowH + rowH / 2, y0 = cy - barH / 2, y1 = cy + barH / 2;
        var x0 = x(floor), x1 = Math.max(x(m.elo), x0 + 4), rad = Math.min(4, x1 - x0);
        var inf = info(m.i), anchor = m.i === ANCHOR;
        var tip = MODELS[m.i] + '\nElo ' + fmtInt(m.elo) + (anchor ? ' (fixed reference)' : ', 95% CI ' + fmtInt(m.lo) + '–' + fmtInt(m.hi)) +
          '\navg. rank ' + m.rank.toFixed(2) + ' · wins ' + m.wins + ' / ' + st.n;
        s += '<g class="e-row" data-tip="' + esc(tip) + '" tabindex="0">';
        s += '<rect class="e-bg" x="0" y="' + (T + r * rowH) + '" width="' + W + '" height="' + rowH + '" rx="4"/>';
        s += '<path d="M' + x0 + ' ' + y0 + 'H' + (x1 - rad) + 'Q' + x1 + ' ' + y0 + ' ' + x1 + ' ' + (y0 + rad) + 'V' + (y1 - rad) + 'Q' + x1 + ' ' + y1 + ' ' + (x1 - rad) + ' ' + y1 + 'H' + x0 + 'Z" style="fill:' + col(m.i) + '"/>';
        if (!anchor) {
          s += '<line class="e-ci" x1="' + x(m.lo) + '" x2="' + x(m.hi) + '" y1="' + cy + '" y2="' + cy + '"/>';
          s += '<line class="e-ci" x1="' + x(m.lo) + '" x2="' + x(m.lo) + '" y1="' + (cy - 5) + '" y2="' + (cy + 5) + '"/>';
          s += '<line class="e-ci" x1="' + x(m.hi) + '" x2="' + x(m.hi) + '" y1="' + (cy - 5) + '" y2="' + (cy + 5) + '"/>';
        }
        s += '<text class="e-val" x="' + (x(Math.max(m.hi, m.elo)) + 8) + '" y="' + (cy + 4) + '">' + fmtInt(m.elo) + '</text>';
        s += '<text class="e-name" x="' + (LW - 12) + '" y="' + (narrow ? cy + 4 : cy - 1) + '" text-anchor="end"' + (narrow ? ' style="font-size:12px"' : '') + '>' + esc(MODELS[m.i]) + '</text>';
        if (!narrow) s += '<text class="e-type" x="' + (LW - 12) + '" y="' + (cy + 13) + '" text-anchor="end">' + esc(anchor ? 'reference · ' + inf.family : inf.family) + '</text>';
        s += '</g>';
      });
      s += '<line x1="' + x(floor) + '" x2="' + x(floor) + '" y1="' + T + '" y2="' + (base + 4) + '" style="stroke:var(--rule-strong)"/>';
      box.innerHTML = s + '</svg>';
    }

    // ---------- Win-rate matrix ----------
    function renderMatrix(st) {
      var box = $('matrix');
      if (!st.n) { box.innerHTML = '<p class="empty">No datasets in this slice.</p>'; return; }
      var order = st.models.slice().sort(function (a, b) { return b.elo - a.elo; }).map(function (m) { return m.i; });
      var LW = 112, TOP = 100, C = 46, W = LW + C * K + 64, H = TOP + C * K + 4;
      var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Pairwise win rates between models">';
      order.forEach(function (j, c) {
        var cx = LW + c * C + C / 2;
        s += '<text class="m-label" transform="translate(' + (cx + 4) + ',' + (TOP - 8) + ') rotate(-40)">' + esc(MODELS[j]) + '</text>';
      });
      order.forEach(function (i, r) {
        var y = TOP + r * C;
        s += '<circle cx="8" cy="' + (y + C / 2) + '" r="4.5" style="fill:' + col(i) + '"/>';
        s += '<text class="m-label" x="18" y="' + (y + C / 2 + 4) + '">' + esc(MODELS[i]) + '</text>';
        order.forEach(function (j, c) {
          var x = LW + c * C;
          var cell = st.wr[i][j];
          if (!cell) {
            s += '<rect x="' + (x + 1) + '" y="' + (y + 1) + '" width="' + (C - 2) + '" height="' + (C - 2) + '" rx="3" style="fill:var(--paper)"/>';
            return;
          }
          var t = (cell.rate - 0.5) * 2, a = Math.round(Math.abs(t) * 100);
          var fill = 'color-mix(in oklab, ' + (t >= 0 ? 'var(--div-pos)' : 'var(--div-neg)') + ' ' + a + '%, var(--div-mid))';
          var ink = Math.abs(t) > 0.55 ? '#ffffff' : 'var(--ink)';
          var tipText = MODELS[i] + ' vs ' + MODELS[j] + '\nlower MSE on ' + cell.wins + ' of ' + st.n + ' datasets (' + Math.round(cell.rate * 100) + '% incl. ties)';
          s += '<rect x="' + (x + 1) + '" y="' + (y + 1) + '" width="' + (C - 2) + '" height="' + (C - 2) + '" rx="3" style="fill:' + fill + '" data-tip="' + esc(tipText) + '" tabindex="0"/>';
          s += '<text class="m-cell-text" x="' + (x + C / 2) + '" y="' + (y + C / 2 + 4) + '" text-anchor="middle" style="fill:' + ink + '">' + Math.round(cell.rate * 100) + '</text>';
        });
      });
      s += '</svg>';
      box.innerHTML = s;
    }

    // ---------- Rank vs complexity small multiples ----------
    function renderTrend(st, idx) {
      var box = $('multiples'), tbl = $('trend-table');
      if (idx.length < 8) {
        box.innerHTML = '<p class="empty" style="grid-column:1/-1">This view needs at least 8 datasets in the slice. The current slice has ' + idx.length + '.</p>';
        tbl.innerHTML = '';
        return;
      }
      var nb = Math.min(5, Math.floor(idx.length / 4));
      var sorted = idx.slice().sort(function (a, b) { return D[a].jgd - D[b].jgd; });
      var bins = range(nb).map(function (g) { return sorted.slice(Math.floor(g * sorted.length / nb), Math.floor((g + 1) * sorted.length / nb)); });
      var med = bins.map(function (b) { var v = b.map(function (d) { return D[d].jgd; }).sort(function (x, y) { return x - y; }); return v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2; });
      var avg = range(K).map(function (i) { return bins.map(function (b) { return mean(b.map(function (d) { return RANK[d][i]; })); }); });
      var order = st.models.slice().sort(function (a, b) { return b.elo - a.elo; }).map(function (m) { return m.i; });
      var VW = 240, VH = 150, L = 22, R = 228, T = 14, Bt = 116;
      function px(g) { return nb === 1 ? (L + R) / 2 : L + 14 + g * (R - L - 28) / (nb - 1); }
      function py(r) { return T + (r - 1) / (K - 1) * (Bt - T); }
      function path(arr) { return arr.map(function (r, g) { return (g ? 'L' : 'M') + px(g).toFixed(1) + ' ' + py(r).toFixed(1); }).join(''); }
      box.innerHTML = order.map(function (i) {
        var s = '<svg viewBox="0 0 ' + VW + ' ' + VH + '" role="img" aria-label="' + esc(MODELS[i]) + ' average rank by JGD group">';
        for (var r = 1; r <= K; r++) {
          s += '<line class="p-grid" x1="' + L + '" x2="' + R + '" y1="' + py(r) + '" y2="' + py(r) + '"/>';
        }
        s += '<text class="p-tick" x="' + (L - 6) + '" y="' + (py(1) + 3.5) + '" text-anchor="end">1</text>';
        s += '<text class="p-tick" x="' + (L - 6) + '" y="' + (py(K) + 3.5) + '" text-anchor="end">' + K + '</text>';
        med.forEach(function (m, g) { s += '<text class="p-tick" x="' + px(g) + '" y="' + (Bt + 18) + '" text-anchor="middle">' + fmtJ(m) + '</text>'; });
        s += '<text class="p-tick" x="' + R + '" y="' + (VH - 2) + '" text-anchor="end">JGD →</text>';
        order.forEach(function (j) { if (j !== i) s += '<path class="p-other" d="' + path(avg[j]) + '"/>'; });
        s += '<path class="p-line" d="' + path(avg[i]) + '" style="stroke:' + col(i) + '"/>';
        avg[i].forEach(function (r, g) {
          s += '<circle class="p-dot" cx="' + px(g) + '" cy="' + py(r) + '" r="4" style="fill:' + col(i) + '"/>';
        });
        [0, nb - 1].forEach(function (g) {
          var r = avg[i][g], above = r > 1.6;
          s += '<text class="p-val" x="' + px(g) + '" y="' + (above ? py(r) - 9 : py(r) + 16) + '" text-anchor="middle">' + r.toFixed(1) + '</text>';
        });
        avg[i].forEach(function (r, g) {
          var tip = MODELS[i] + '\nJGD ' + fmtJ(D[bins[g][0]].jgd) + '–' + fmtJ(D[bins[g][bins[g].length - 1]].jgd) + ' · ' + bins[g].length + ' datasets\naverage rank ' + r.toFixed(2);
          s += '<circle class="p-hit" cx="' + px(g) + '" cy="' + py(r) + '" r="12" data-tip="' + esc(tip) + '" tabindex="0"/>';
        });
        s += '</svg>';
        return '<figure class="panel"><figcaption><span class="sw" style="background:' + col(i) + '"></span>' + esc(MODELS[i]) + '</figcaption>' + s + '</figure>';
      }).join('');
      tbl.innerHTML = '<table><thead><tr><th scope="col">Model</th>' + med.map(function (m, g) { return '<th scope="col" class="num">Group ' + (g + 1) + '<br><span class="hint">JGD ≈ ' + fmtJ(m) + '</span></th>'; }).join('') + '</tr></thead><tbody>' +
        order.map(function (i) { return '<tr><td>' + esc(MODELS[i]) + '</td>' + avg[i].map(function (r) { return '<td class="num">' + r.toFixed(2) + '</td>'; }).join('') + '</tr>'; }).join('') + '</tbody></table>';
    }

    // ---------- Per-dataset table ----------
    function renderDatasets(idx) {
      var box = $('ds-table');
      if (!idx.length) { box.innerHTML = '<p class="empty">No datasets match this slice.</p>'; return; }
      var key = state.dsSort.key, dir = state.dsSort.dir;
      var rows = idx.slice().sort(function (a, b) {
        var va, vb;
        if (key === 'id') { va = D[a].id; vb = D[b].id; return dir * va.localeCompare(vb); }
        if (key === 'domain') { va = D[a].domain; vb = D[b].domain; return dir * va.localeCompare(vb) || D[b].jgd - D[a].jgd; }
        if (key === 'ch') return dir * (D[a].channels - D[b].channels) || D[b].jgd - D[a].jgd;
        if (key === 'jgd') return dir * (D[a].jgd - D[b].jgd);
        var m = +key.slice(1);
        return dir * (MSE[a][m] / BEST[a] - MSE[b][m] / BEST[b]);
      });
      var lmin = Math.log10(Math.min.apply(null, D.map(function (d) { return d.jgd; })));
      var lmax = Math.log10(Math.max.apply(null, D.map(function (d) { return d.jgd; })));
      function th(k, label, cls) {
        var sortAttr = key === k ? ' aria-sort="' + (dir < 0 ? 'descending' : 'ascending') + '"' : '';
        return '<th scope="col"' + (cls ? ' class="' + cls + '"' : '') + sortAttr + '><button class="sort" type="button" data-dsort="' + k + '">' + label + '</button></th>';
      }
      var h = '<table class="ds"><thead><tr>' + th('id', 'Dataset', 'sticky') + th('ch', 'Ch.', 'num') + th('jgd', 'JGD', 'num') +
        MODELS.map(function (m, i) { return th('m' + i, '<span class="sw" style="background:' + col(i) + '"></span>' + esc(m), 'num'); }).join('') + '</tr></thead><tbody>';
      h += rows.map(function (d) {
        var ds = D[d];
        var w = Math.max(4, (Math.log10(ds.jgd) - lmin) / (lmax - lmin) * 100);
        var detail = ds.model + '\n' + prettyDomain(ds.domain) + ' · ' + ds.channels + ' channels · ' + ds.constants + ' constants\nT = ' + ds.T + ' · σ_states ' + ds.ds + ' · σ_const ' + ds.dc;
        var r = '<tr><td class="sticky" data-tip="' + esc(detail) + '"><button type="button" class="ds-open" data-explore="' + esc(ds.id) + '">' + esc(ds.id) + '</button><span class="ds-dom">' + esc(prettyDomain(ds.domain)) + '</span><span class="ds-model">' + esc(ds.model) + '</span></td>' +
          '<td class="num">' + ds.channels + '</td>' +
          '<td class="num"><div class="jgd-cell">' + fmtJ(ds.jgd) + '<span class="jgd-bar" aria-hidden="true"><i style="width:' + w.toFixed(0) + '%"></i></span></div></td>';
        r += MODELS.map(function (_, i) {
          var ratio = MSE[d][i] / BEST[d];
          var close = Math.max(0, 1 - Math.log(ratio) / Math.log(4));
          var pct = Math.round(close * close * 34);
          var best = MSE[d][i] === BEST[d];
          var bg = pct > 0 ? ' style="background:color-mix(in oklab, var(--seq) ' + pct + '%, transparent)"' : '';
          var val = state.view === 'mse'
            ? '<span class="v">' + MSE[d][i].toFixed(3) + '</span><span class="sd">± ' + SD[d][i].toFixed(3) + '</span>'
            : '<span class="v">' + fmtRatio(ratio) + '</span>';
          return '<td class="res' + (best ? ' best' : '') + '"' + bg + '>' + val + '</td>';
        }).join('');
        return r + '</tr>';
      }).join('');
      box.innerHTML = h + '</tbody></table>';
    }

    // ---------- Datasets section ----------
    var DOMAINS = (function () {
      var c = {};
      D.forEach(function (d) { c[d.domain] = (c[d.domain] || 0) + 1; });
      return Object.keys(c).map(function (k) { return { k: k, n: c[k] }; }).sort(function (a, b) { return b.n - a.n || a.k.localeCompare(b.k); });
    })();

    function renderDomains() {
      var max = DOMAINS[0].n;
      $('domains').innerHTML = DOMAINS.map(function (d) {
        return '<button type="button" class="dom-row" data-domain="' + esc(d.k) + '" aria-pressed="' + (state.domain === d.k) + '"><span class="dom-name">' + esc(prettyDomain(d.k)) + '</span>' +
          '<span class="dom-bar"><i style="width:' + (d.n / max * 82).toFixed(1) + '%"></i>' + d.n + '</span></button>';
      }).join('');
    }

    var STRIP = (function () {
      var W = 560, L = 16, R = 544, base = 112, r = 4.5;
      var lmin = Math.log10(0.005), lmax = Math.log10(3);
      function x(v) { return L + (Math.log10(v) - lmin) / (lmax - lmin) * (R - L); }
      var placed = [];
      var pts = byJgd.slice().reverse().map(function (d) {
        var px = x(D[d].jgd), level = 0;
        while (placed.some(function (p) { return p.level === level && Math.abs(p.x - px) < 2 * r + 1; })) level++;
        var p = { d: d, x: px, level: level };
        placed.push(p);
        return p;
      });
      return { W: W, x: x, pts: pts, base: base, r: r, L: L, R: R };
    })();

    function renderStrip(idx) {
      var inSlice = {}; idx.forEach(function (d) { inSlice[d] = true; });
      var s = '';
      var ticks = [0.01, 0.03, 0.1, 0.3, 1, 3];
      s += '<line x1="' + STRIP.L + '" x2="' + STRIP.R + '" y1="' + (STRIP.base + 8) + '" y2="' + (STRIP.base + 8) + '" style="stroke:var(--rule-strong)"/>';
      ticks.forEach(function (t) {
        s += '<text class="axis-label" x="' + STRIP.x(t) + '" y="' + (STRIP.base + 24) + '" text-anchor="middle">' + t + '</text>';
      });
      var hiMin = Math.min.apply(null, byJgd.slice(0, third).map(function (d) { return D[d].jgd; }));
      var midMax = D[byJgd[third]].jgd;
      var midMin = Math.min.apply(null, byJgd.slice(third, 2 * third).map(function (d) { return D[d].jgd; }));
      var lowMax = D[byJgd[2 * third]].jgd;
      var c1 = STRIP.x(Math.sqrt(hiMin * midMax)), c2 = STRIP.x(Math.sqrt(midMin * lowMax));
      [c1, c2].forEach(function (c) { s += '<line class="s-cut" x1="' + c + '" x2="' + c + '" y1="6" y2="' + (STRIP.base + 8) + '"/>'; });
      s += '<text class="s-band" x="' + ((STRIP.L + c2) / 2) + '" y="14" text-anchor="middle">Low</text>';
      s += '<text class="s-band" x="' + ((c1 + c2) / 2) + '" y="14" text-anchor="middle">Medium</text>';
      s += '<text class="s-band" x="' + ((c1 + STRIP.R) / 2) + '" y="14" text-anchor="middle">High</text>';
      STRIP.pts.forEach(function (p) {
        var ds = D[p.d], cy = STRIP.base - p.level * (2 * STRIP.r + 1);
        s += '<circle class="s-dot ' + (inSlice[p.d] ? 'in' : 'out') + '" cx="' + p.x.toFixed(1) + '" cy="' + cy + '" r="' + STRIP.r + '" data-explore="' + esc(ds.id) + '" data-tip="' + esc(ds.id + ' · ' + ds.model + '\nJGD ' + fmtJ(ds.jgd)) + '"/>';
      });
      s += '<text class="axis-label" x="' + STRIP.R + '" y="' + (STRIP.base + 38) + '" text-anchor="end">JGD, log scale</text>';
      $('strip').innerHTML = s;
    }

    // ---------- Controls ----------
    function tierRange(t) {
      var v = range(N_ALL).filter(function (i) { return TIER[i] === t; }).map(function (i) { return D[i].jgd; });
      return fmtJ(Math.min.apply(null, v)) + '–' + fmtJ(Math.max.apply(null, v));
    }
    document.querySelectorAll('[data-ctl="tier"] button').forEach(function (b) {
      if (b.dataset.v !== 'all') b.title = 'JGD ' + tierRange(b.dataset.v);
    });
    var sel = $('f-domain');
    sel.innerHTML = '<option value="all">All domains</option>' + DOMAINS.map(function (d) { return '<option value="' + esc(d.k) + '">' + esc(prettyDomain(d.k)) + ' (' + d.n + ')</option>'; }).join('');

    function renderControls(idx) {
      document.querySelectorAll('.seg').forEach(function (seg) {
        var k = seg.dataset.ctl;
        seg.querySelectorAll('button').forEach(function (b) { b.setAttribute('aria-pressed', String(state[k] === b.dataset.v)); });
      });
      sel.value = state.domain;
      var n = idx.length, txt = '<strong>' + n + '</strong> of ' + N_ALL + ' datasets';
      if (n) {
        var j = idx.map(function (d) { return D[d].jgd; });
        txt += ' · JGD ' + fmtJ(Math.min.apply(null, j)) + '–' + fmtJ(Math.max.apply(null, j));
      }
      if (n && n < 10) txt += ' · <span class="warn">few datasets, wide intervals</span>';
      $('slice-summary').innerHTML = txt;
      $('reset').hidden = state.tier === 'all' && state.domain === 'all' && state.ch === 'all';
    }

    function render() {
      var idx = sliceIdx();
      var st = idx.length ? computeStats(idx) : { n: 0, models: [], wr: [] };
      lastStats = st;
      renderControls(idx);
      renderEloChart(st);
      renderLeaderboard(st);
      renderMatrix(st);
      if (idx.length) renderTrend(st, idx); else { $('multiples').innerHTML = '<p class="empty" style="grid-column:1/-1">No datasets in this slice.</p>'; $('trend-table').innerHTML = ''; }
      renderDatasets(idx);
      renderDomains();
      renderStrip(idx);
    }

    document.querySelectorAll('.seg').forEach(function (seg) {
      seg.addEventListener('click', function (e) {
        var b = e.target.closest('button');
        if (!b) return;
        state[seg.dataset.ctl] = b.dataset.v;
        if (seg.dataset.ctl === 'view') { renderControls(sliceIdx()); renderDatasets(sliceIdx()); return; }
        render();
      });
    });
    sel.addEventListener('change', function () { state.domain = sel.value; render(); });
    $('reset').addEventListener('click', function () { state.tier = 'all'; state.domain = 'all'; state.ch = 'all'; render(); });
    document.querySelector('.lb thead').addEventListener('click', function (e) {
      var b = e.target.closest('[data-sort]');
      if (!b) return;
      var k = b.dataset.sort;
      state.lbSort = state.lbSort.key === k ? { key: k, dir: -state.lbSort.dir } : { key: k, dir: DEFAULT_DIR[k] };
      renderLeaderboard(computeStatsCached());
    });
    $('ds-table').addEventListener('click', function (e) {
      var b = e.target.closest('[data-dsort]');
      if (!b) return;
      var k = b.dataset.dsort;
      var def = k === 'id' || k === 'domain' || k.charAt(0) === 'm' ? 1 : -1;
      state.dsSort = state.dsSort.key === k ? { key: k, dir: -state.dsSort.dir } : { key: k, dir: def };
      renderDatasets(sliceIdx());
    });
    $('domains').addEventListener('click', function (e) {
      var b = e.target.closest('[data-domain]');
      if (!b) return;
      state.domain = state.domain === b.dataset.domain ? 'all' : b.dataset.domain;
      render();
      $('leaderboard').scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
    });

    var cache = { key: null, st: null };
    function computeStatsCached() {
      var idx = sliceIdx(), k = idx.join(',');
      if (cache.key !== k) cache = { key: k, st: idx.length ? computeStats(idx) : { n: 0, models: [], wr: [] } };
      return cache.st;
    }

    // ---------- Tooltip ----------
    var tip = $('tip');
    function place(x, y) {
      var r = tip.getBoundingClientRect();
      var left = x + 14, top = y + 16;
      if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
      if (top + r.height > window.innerHeight - 8) top = y - r.height - 12;
      tip.style.left = Math.max(8, left) + 'px';
      tip.style.top = Math.max(8, top) + 'px';
    }
    document.addEventListener('pointermove', function (e) {
      var t = e.target.closest && e.target.closest('[data-tip]');
      if (!t) { tip.hidden = true; return; }
      tip.textContent = t.getAttribute('data-tip');
      tip.hidden = false;
      place(e.clientX, e.clientY);
    });
    document.addEventListener('focusin', function (e) {
      var t = e.target.closest && e.target.closest('[data-tip]');
      if (!t) { tip.hidden = true; return; }
      tip.textContent = t.getAttribute('data-tip');
      tip.hidden = false;
      var r = t.getBoundingClientRect();
      place(r.right, r.top);
    });
    document.addEventListener('focusout', function () { tip.hidden = true; });
    window.addEventListener('scroll', function () { tip.hidden = true; }, { passive: true });

    // ---------- Copy BibTeX ----------
    $('copy-bib').addEventListener('click', function () {
      var btn = this, text = $('bibtex').textContent;
      function selectIt() {
        var range = document.createRange(); range.selectNodeContents($('bibtex'));
        var s = window.getSelection(); s.removeAllRanges(); s.addRange(range);
        btn.textContent = 'Press Ctrl+C';
      }
      try {
        navigator.clipboard.writeText(text).then(function () { btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = 'Copy'; }, 1600); }, selectIt);
      } catch (err) { selectIt(); }
    });

    // ---------- Samples: benchmark instances regenerated with the benchmark's seeds ----------
    var DS_BY_ID = {};
    D.forEach(function (d, i) { DS_BY_ID[d.id] = i; });
    var SAMPLES = {};
    var KEEP = 0.2, NOISE = 0.05, LANE_LIMIT = 6;

    function fmtScore(x) { return x < 1e-4 ? '< 0.0001' : fmtJ(x); }
    function fmtT(x) { return x === 0 ? '0' : x >= 10 ? x.toFixed(1) : x >= 1 ? x.toFixed(2) : x.toPrecision(2); }
    function shortUnit(u) {
      var m = { second: 's', s: 's', minute: 'min', min: 'min', millisecond: 'ms', ms: 'ms', hour: 'h', day: 'd', dimensionless: '' };
      return Object.prototype.hasOwnProperty.call(m, u) ? m[u] : u;
    }
    function reduceMotion() { return window.matchMedia('(prefers-reduced-motion: reduce)').matches; }

    // What a model sees for one instance: fresh noise on the z-scored values and an independent 20% keep-mask per value (80% of observations are dropped).
    function drawInstance(sample, inst, seed) {
      var rng = mulberry32(seed);
      function gauss() { var u = 1 - rng(), v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
      return sample.z[inst].map(function (series) {
        return series.map(function (v) { return { v: v + NOISE * gauss(), keep: rng() < KEEP }; });
      });
    }

    function laneSVG(sample, inst, c, drawn, o) {
      var W = o.w, H = o.h, L = 8, R = W - 8, T = o.top || 8, B = H - 20;
      var t = sample.t, tmax = t[t.length - 1], tObs = tmax / 2;
      function x(v) { return L + v / tmax * (R - L); }
      var truth = sample.z[inst][c], pts = drawn[c];
      var vals = truth.slice();
      pts.forEach(function (p) { if (p.keep) vals.push(p.v); });
      var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
      var mid = (lo + hi) / 2, half = Math.max((hi - lo) / 2, 0.3) * 1.15;
      function y(v) { return T + (mid + half - v) / (2 * half) * (B - T); }
      var ch = sample.channels[c];
      var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + esc(ch.name) + ': observed values in the first half, forecast targets in the second half">';
      var step = (R - L) / 40, i;
      for (i = 0; i <= 40; i++) {
        var gx = (L + i * step).toFixed(1);
        s += '<line class="' + (i % 5 ? 'g-minor' : 'g-major') + '" x1="' + gx + '" x2="' + gx + '" y1="' + T + '" y2="' + B + '"/>';
      }
      for (i = 0; B - i * step >= T - 0.1; i++) {
        var gy = (B - i * step).toFixed(1);
        s += '<line class="' + (i % 5 ? 'g-minor' : 'g-major') + '" x1="' + L + '" x2="' + R + '" y1="' + gy + '" y2="' + gy + '"/>';
      }
      s += '<rect x="' + x(tObs) + '" y="' + T + '" width="' + (R - x(tObs)) + '" height="' + (B - T) + '" style="fill:var(--accent-soft)"/>';
      s += '<line x1="' + x(tObs) + '" x2="' + x(tObs) + '" y1="' + (o.split ? T - 14 : T) + '" y2="' + B + '" style="stroke:var(--accent);stroke-width:1.25"/>';
      if (o.split) {
        s += '<text class="split-label" x="' + (x(tObs) - 8) + '" y="' + (T - 6) + '" text-anchor="end" style="fill:var(--ink-2)">← observed</text>';
        s += '<text class="split-label" x="' + (x(tObs) + 8) + '" y="' + (T - 6) + '" style="fill:var(--accent)">forecast →</text>';
      }
      if (o.truth) {
        s += '<path d="' + truth.map(function (v, k) { return (k ? 'L' : 'M') + x(t[k]).toFixed(1) + ' ' + y(v).toFixed(1); }).join('') + '" style="fill:none;stroke:var(--ink-2);stroke-opacity:0.4;stroke-width:1.5;stroke-linejoin:round"/>';
      }
      pts.forEach(function (p, k) {
        if (!p.keep || t[k] >= tmax) return;
        var cx = x(t[k]).toFixed(1), cy = y(p.v).toFixed(1);
        if (t[k] <= tObs) s += '<circle cx="' + cx + '" cy="' + cy + '" r="3.8" style="fill:var(--ink);stroke:var(--surface);stroke-width:1.5"/>';
        else s += '<circle cx="' + cx + '" cy="' + cy + '" r="3.6" style="fill:var(--surface);stroke:var(--accent);stroke-width:1.6"/>';
      });
      if (o.label) s += '<text class="lane-label" x="' + (L + 6) + '" y="' + (T + 15) + '">' + esc(o.label) + '</text>';
      var u = shortUnit(sample.timeUnit);
      [0, tObs, tmax].forEach(function (v, k) {
        s += '<text class="axis-label" x="' + x(v).toFixed(1) + '" y="' + (B + 15) + '" text-anchor="' + ['start', 'middle', 'end'][k] + '">' + fmtT(v) + (k === 2 && u ? ' ' + esc(u) : '') + '</text>';
      });
      return s + '</svg>';
    }

    function byScore(sample) {
      return range(sample.channels.length).sort(function (a, b) { return sample.channels[b].score - sample.channels[a].score; });
    }

    function renderHero(h) {
      var inst = Math.max(0, h.instances.indexOf(HERO.instance));
      var drawn = drawInstance(h, inst, HERO.seed);
      $('hero-lanes').innerHTML = byScore(h).slice(0, 3).map(function (c, k) {
        return laneSVG(h, inst, c, drawn, { w: 600, h: k === 0 ? 134 : 110, top: k === 0 ? 30 : 6, split: k === 0, truth: true, label: h.channels[c].name });
      }).join('');
      $('hero-cap').innerHTML = '<strong>' + esc(h.id) + '</strong> (<span class="mono">' + esc(h.model) + '</span>), benchmark instance ' + h.instances[inst] + ', with the benchmark’s 20% sampling and σ = 0.05 noise.';
    }

    var ex = { id: null, inst: 0, seed: 1, all: false, truth: true, data: null };
    var exSel = $('ex-ds');
    exSel.innerHTML = DOMAINS.map(function (dom) {
      return '<optgroup label="' + esc(prettyDomain(dom.k)) + '">' + D.filter(function (d) { return d.domain === dom.k; })
        .sort(function (a, b) { return b.jgd - a.jgd; })
        .map(function (d) { return '<option value="' + esc(d.id) + '">' + esc(d.id) + ' · ' + esc(d.model) + '</option>'; }).join('') + '</optgroup>';
    }).join('');

    function loadSample(id) {
      if (SAMPLES[id]) return Promise.resolve(SAMPLES[id]);
      return fetch('samples/' + id + '.json').then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      }).then(function (j) { SAMPLES[id] = j; return j; });
    }

    function renderExMeta() {
      var i = DS_BY_ID[ex.id], d = D[i];
      var bests = range(K).filter(function (m) { return MSE[i][m] === BEST[i]; });
      var bestTxt = '<span class="sw" style="background:' + col(bests[0]) + '"></span>' + esc(MODELS[bests[0]]) +
        (bests.length === 2 ? ' / ' + esc(MODELS[bests[1]]) : bests.length > 2 ? ' + ' + (bests.length - 1) + ' tied' : '') +
        ' <span class="of">' + BEST[i].toFixed(3) + '</span>';
      var items = [
        ['Physiome model', '<span class="mono">' + esc(d.model) + '</span>'],
        ['Domain', esc(prettyDomain(d.domain))],
        ['Size', d.channels + ' states · ' + d.constants + ' constants'],
        ['JGD', fmtJ(d.jgd) + ' · ' + { high: 'high', mid: 'medium', low: 'low' }[TIER[i]]],
        ['σ_states / σ_const', d.ds + ' / ' + d.dc],
        ['Best model (MSE)', bestTxt]
      ];
      $('ex-meta').innerHTML = items.map(function (p) { return '<div><dt>' + p[0] + '</dt><dd>' + p[1] + '</dd></div>'; }).join('');
    }

    function renderExplorer() {
      var data = ex.data, box = $('ex-lanes');
      box.classList.remove('loading');
      var drawn = drawInstance(data, ex.inst, ex.seed);
      var order = byScore(data);
      var shown = ex.all ? order : order.slice(0, LANE_LIMIT);
      var t = data.t, tmax = t[t.length - 1], nObs = 0, nTgt = 0;
      drawn.forEach(function (pts) {
        pts.forEach(function (p, k) { if (!p.keep) return; if (t[k] <= tmax / 2) nObs++; else if (t[k] < tmax) nTgt++; });
      });
      box.innerHTML = shown.map(function (c) {
        var ch = data.channels[c];
        var sub = [ch.comp !== ch.name ? ch.comp : '', ch.unit && ch.unit !== 'dimensionless' ? ch.unit : ''].filter(Boolean).join(' · ');
        return '<figure class="lane"><figcaption class="lane-head"><span class="lane-name">' + esc(ch.name) + '</span><span class="lane-comp">' + esc(sub) + '</span><span class="lane-score">JGD ' + fmtScore(ch.score) + '</span></figcaption>' +
          laneSVG(data, ex.inst, c, drawn, { w: 420, h: 128, truth: ex.truth }) + '</figure>';
      }).join('');
      $('ex-status').innerHTML = '<strong>' + esc(data.id) + '</strong>, instance ' + data.instances[ex.inst] + ': ' + nObs + ' observed values and ' + nTgt + ' forecast targets across ' + data.channels.length + ' channels' +
        (shown.length < data.channels.length ? '. Showing the ' + shown.length + ' channels with the highest JGD.' : '.');
      var more = $('ex-more');
      more.hidden = data.channels.length <= LANE_LIMIT;
      more.textContent = ex.all ? 'Show top ' + LANE_LIMIT + ' channels' : 'Show all ' + data.channels.length + ' channels';
    }

    function resample(fresh) {
      var n = ex.data.z.length, next = Math.floor(Math.random() * n);
      if (!fresh && n > 1 && next === ex.inst) next = (next + 1) % n;
      ex.inst = next;
      ex.seed = Math.floor(Math.random() * 2147483647);
      renderExplorer();
    }

    function openExplorer(id, scroll) {
      if (!Object.prototype.hasOwnProperty.call(DS_BY_ID, id)) return;
      ex.id = id; ex.all = false; exSel.value = id;
      renderExMeta();
      $('ex-lanes').classList.add('loading');
      $('ex-status').textContent = 'Loading ' + id + '…';
      loadSample(id).then(function (data) {
        if (ex.id !== id) return;
        ex.data = data;
        resample(true);
      }, function () {
        if (ex.id !== id) return;
        ex.data = null;
        $('ex-lanes').classList.remove('loading');
        $('ex-lanes').innerHTML = '<p class="empty">Could not load samples/' + esc(id) + '.json. The sample files are published next to this page; check that the samples folder is there.</p>';
        $('ex-status').textContent = '';
        $('ex-more').hidden = true;
      });
      if (scroll) $('explore').scrollIntoView({ behavior: reduceMotion() ? 'auto' : 'smooth', block: 'start' });
    }

    exSel.addEventListener('change', function () { openExplorer(exSel.value, false); });
    $('ex-sample').addEventListener('click', function () { if (ex.data) resample(false); });
    $('ex-truth').addEventListener('change', function () { ex.truth = this.checked; if (ex.data) renderExplorer(); });
    $('ex-more').addEventListener('click', function () { ex.all = !ex.all; renderExplorer(); });
    document.addEventListener('click', function (e) {
      var t = e.target.closest && e.target.closest('[data-explore]');
      if (t) openExplorer(t.getAttribute('data-explore'), true);
    });

    var eloWidth = 0;
    if (window.ResizeObserver) {
      new ResizeObserver(function () {
        var w = $('elo-chart').clientWidth;
        if (lastStats && Math.abs(w - eloWidth) > 1) renderEloChart(lastStats);
      }).observe($('elo-chart'));
    }

    var chans = D.map(function (d) { return d.channels; });
    $('fact-datasets').textContent = N_ALL;
    $('fact-domains').textContent = DOMAINS.length;
    $('fact-channels').textContent = Math.min.apply(null, chans) + '–' + Math.max.apply(null, chans);
    $('fact-models').textContent = K;
    $('updated').textContent = 'Leaderboard data built ' + RAW.generated;
    loadSample(HERO.id).then(renderHero, function () { $('hero-lanes').closest('figure').hidden = true; });
    render();
    openExplorer(EXPLORER_DEFAULT, false);
  }
})();
