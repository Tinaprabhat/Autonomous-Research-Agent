"""
dashboard.py — bakes eval/reports/latest.json (+ query-log latency stats) into one
self-contained static page: eval/reports/dashboard.html. No server, no build step.

    python -m eval.dashboard
"""
from __future__ import annotations

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

HERE = os.path.dirname(__file__)

TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Eval Dashboard</title>
<style>
:root{--bg:#0d0f0f;--s:#131616;--b:#2a3030;--t:#e8eded;--m:#7d8c8c;--a:#4fffb0;--w:#ffcc4a;--e:#ff6b4a}
*{box-sizing:border-box}body{background:var(--bg);color:var(--t);font:15px/1.5 'DM Sans',system-ui,sans-serif;margin:0;padding:32px 20px}
main{max-width:980px;margin:0 auto}h1{font-weight:500;margin:0 0 4px}h2{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--a);margin:34px 0 10px;font-family:monospace}
.sub{color:var(--m);margin-bottom:20px}table{width:100%;border-collapse:collapse;background:var(--s);border:1px solid var(--b);border-radius:8px;overflow:hidden}
th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--b);font-variant-numeric:tabular-nums}th:first-child,td:first-child{text-align:left}
th{color:var(--m);font-weight:500;font-size:12px}tr:last-child td{border-bottom:0}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.card{background:var(--s);border:1px solid var(--b);border-radius:8px;padding:12px 14px}.card b{display:block;font-size:26px;font-weight:500}.card span{color:var(--m);font-size:12px}
.bar{height:6px;background:var(--b);border-radius:3px;margin-top:6px}.bar i{display:block;height:100%;background:var(--a);border-radius:3px}
.empty{color:var(--m);font-style:italic}
</style></head><body><main>
<h1>Evaluation dashboard</h1><div class="sub" id="sub"></div>
<div class="cards" id="cards"></div>
<h2>Retrieval ablation</h2><div id="retr"></div>
<h2>Generation quality (DeepEval, Groq judge)</h2>
<div class="sub" id="gennote"></div><div id="gen"></div>
<h2>Answerable vs unanswerable</h2><div id="beh"></div>
<h2>Serving latency &amp; cost</h2><div id="ops"></div>
</main>
<script>
const R = __REPORT__, OPS = __OPS__;
const $ = id => document.getElementById(id), f = v => v == null ? '–' : (typeof v === 'number' ? v.toFixed(3) : v);
function table(rows, cols, firstLabel) {
  const t = document.createElement('table'), h = t.createTHead().insertRow();
  [firstLabel, ...cols].forEach(c => { const th = document.createElement('th'); th.textContent = c; h.appendChild(th); });
  const b = t.createTBody();
  Object.entries(rows).forEach(([name, r]) => { const tr = b.insertRow(); tr.insertCell().textContent = name;
    cols.forEach(c => tr.insertCell().textContent = f(r[c])); });
  return t;
}
function card(label, v, frac) { const d = document.createElement('div'); d.className = 'card';
  const b = document.createElement('b'); b.textContent = f(v); const s = document.createElement('span'); s.textContent = label;
  d.append(b, s); if (frac != null) { const bar = document.createElement('div'); bar.className = 'bar';
    const i = document.createElement('i'); i.style.width = Math.max(0, Math.min(1, frac)) * 100 + '%'; bar.appendChild(i); d.appendChild(bar); } return d; }
if (!R) { $('sub').textContent = 'No report yet — run: python -m eval.runner all'; }
else {
  $('sub').textContent = R.timestamp + ' · ' + R.n_cases + ' cases (' + R.n_answerable + ' answerable, ' + R.n_unanswerable + ' unanswerable) · k=' + R.k;
  if (R.composite != null) $('cards').appendChild(card('Composite (faith · relevancy · citation)', R.composite, R.composite));
  if (R.retrieval) { const hr = R.retrieval['hybrid+rerank']; const k = R.k;
    $('cards').appendChild(card('Recall@' + k + ' (hybrid+rerank)', hr['chunk_recall@' + k], hr['chunk_recall@' + k]));
    $('cards').appendChild(card('MRR (hybrid+rerank)', hr.chunk_mrr, hr.chunk_mrr));
    const cols = Object.keys(hr).filter(c => c !== 'n'); $('retr').appendChild(table(R.retrieval, cols, 'config')); }
  else $('retr').innerHTML = '<span class="empty">not run</span>';
  if (R.generation) { const g0 = R.generation; $('gennote').textContent = 'Scored on cases the guarded pipeline answered (abstentions are reported below as a recall cost, not as unfaithful answers). ' + (g0.cases_scored != null ? g0.cases_scored + ' of ' + g0.cases_planned + ' golden cases evaluated (' + (g0.answerable_scored_with_judge != null ? g0.answerable_scored_with_judge : '?') + ' answerable scored with DeepEval; unanswerable measure abstention only)' : '') + (g0.stopped_early ? ' — stopped early: ' + g0.stopped_early : '') + '.';
    const m = R.generation.metrics, cols = [...new Set(Object.values(m).flatMap(o => Object.keys(o)))].filter(c => c !== 'n');
    $('gen').appendChild(table(m, cols, 'config')); $('beh').appendChild(table({ 'behaviour': R.generation.behaviour },
      ['answer_rate_on_answerable', 'abstention_rate_on_unanswerable', 'answerable_total', 'unanswerable_total'], '')); }
  else { $('gen').innerHTML = '<span class="empty">not run</span>'; $('beh').innerHTML = '<span class="empty">not run</span>'; }
}
if (OPS && OPS.queries) { $('ops').appendChild(table({ 'live traffic': OPS }, ['queries', 'latency_ms_p50', 'latency_ms_p95', 'avg_cost_usd', 'total_cost_usd'], '')); }
else $('ops').innerHTML = '<span class="empty">no queries logged yet</span>';
</script></body></html>
"""


def build(report_path: str | None = None, out_path: str | None = None) -> str:
    from backend.app.telemetry import observability
    report_path = report_path or os.path.join(HERE, "reports", "latest.json")
    out_path = out_path or os.path.join(HERE, "reports", "dashboard.html")
    report = None
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
    # '<' escaped so embedded JSON can never close the script tag
    safe = lambda o: json.dumps(o).replace("<", "\\u003c")
    html = TEMPLATE.replace("__REPORT__", safe(report)).replace("__OPS__", safe(observability.summarize()))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


if __name__ == "__main__":
    print("wrote", build())
